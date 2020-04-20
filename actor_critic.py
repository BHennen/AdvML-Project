import tensorflow as tf
from keras import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import RandomNormal

import numpy as np

class ActorCritic(object):
    ''' Advantage Actor Critic that is updated in batches. Needs model supplied to it.

    '''
    def __init__(self, env, action_values, model, optimizer, learning_rate, gamma, n_max_steps, n_episodes, frameskip=1, reward_fn=None):
        '''

        :param frameskip: Allows the agent to skip n-1 simulations, repeating the chosen action for n frames.
        :param n_max_steps: Number of iterations in the environment before terminating the episode
        :param n_episodes: Number of episodes to run through before updating gradient
        '''
        self.model = model
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optimizer
        self.action_values = action_values
        self.reward_fn = reward_fn
        self.frameskip = frameskip
        self.n_max_steps = n_max_steps
        self.n_max_episodes = n_episodes
        self.cur_episode = 0
        self.cur_step = 0
        self.episode_rewards = []
        self.all_rewards = []
        self.episode_grads = []
        self.all_grads = []
        self.obs = self.env.reset()
        self.next_value = None
        self.next_action_dists = None

    @staticmethod
    def build_model(input_shape, do_dropout=True, p_dropout=0.5, l2_reg_val=0.001, print_summary=False):
        input_tensor = Input(shape=input_shape, name='input')
        x = input_tensor
        if do_dropout:
            x = Dropout(p_dropout)(input_tensor)
        # Convolution layers
        convs = [(15, (7, 7), 3, 'conv1'), (15, (5, 5), 2, 'conv2'), (15, (3, 3), 1, 'conv3'), (15, (3, 3), 1, 'conv4')]
        for (filters, kernel, strides, name) in convs:
            x = Conv2D(filters, kernel_size=kernel, strides=strides, name=name)(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten(name='flatten')(x)
        # Dense layers
        x = Dense(24, activation="relu", name='dense_1', kernel_initializer=RandomNormal(), kernel_regularizer=l2(l2_reg_val))(x)
        output0 = Dense(1, name='value', activation='linear', kernel_initializer=RandomNormal())(x)
        output1 = Dense(3, name='steer', activation='softmax', kernel_initializer=RandomNormal())(x)
        output2 = Dense(2, name='gas', activation='softmax', kernel_initializer=RandomNormal())(x)
        output3 = Dense(2, name='brake', activation='softmax', kernel_initializer=RandomNormal())(x)
        model = Model(inputs=[input_tensor], outputs=[output0, output1, output2, output3], name='agent_policy')
        if print_summary:
            model.summary()
        return model

    def play_one_step(self, obs, value=None, action_dists=None):
        ''' Plays one step of the environment and returns next_obs, reward, done, next_value, next_action_dists, grads
        '''
        with tf.GradientTape() as tape:
            # Get the estimated value and probability distributions for the actions
            if value is None:
                value, *action_dists = self.model(obs)

            # get log probabilities of the distributions
            log_dists = [tf.math.log([action_dist]) for action_dist in action_dists]
            
            # select action indices from the log distributions
            action_idxs = [tf.random.categorical(log_dist,1)[0][0] for log_dist in log_dists]

            # get log probabilities of the actions
            log_probas = [log_dists[which_action][0][action_idx] for which_action, action_idx in enumerate(action_idxs)]
            
            with tape.stop_recording():
                # Get the actual actions taken
                actions = [self.action_values[which_action][action_idx] for which_action, action_idx in enumerate(action_idxs)]
                # and take n steps in the environment
                for _ in range(self.frameskip):
                    next_obs, reward, done, _ = self.env.step(actions)
                    if done:
                        break
                if self.reward_fn:
                    reward = self.reward_fn(next_obs)

            next_value, *next_action_dists = self.model(next_obs)
            advantage = reward + (1.0 - done) * self.gamma * next_value - value
            # TODO: add entropy to actor loss to encourage exploration
            critic_loss = tf.math.pow(advantage, 2)
            actor_loss = tf.reduce_sum(-1 * tf.multiply(log_probas, advantage))
            model_loss = self.model.losses
            total_loss = tf.math.add_n([critic_loss, actor_loss] + model_loss)
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        return next_obs, reward, done, next_value, next_action_dists, grads
    
    def discount_rewards(self, rewards):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1 , -1):
            discounted[step] += discounted[step + 1] * self.gamma
        return discounted
    
    def discount_and_normalize_rewards(self, all_rewards):
        all_discounted_rewards = [self.discount_rewards(rewards) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    def apply_grads(self, all_rewards, all_grads):
        ''' Updates the model according to the rewards we've gathered and the gradients accumulated.
        '''
        normalized_rewards = self.discount_and_normalize_rewards(all_rewards)

        all_mean_grads = []
        for var_index in range(len(self.model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                for episode_index, final_rewards in enumerate(normalized_rewards)
                for step, final_reward in enumerate(final_rewards)],
                axis = 0)
            all_mean_grads.append(mean_grads)
        self.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))

    def training_loop(self):
        ''' Training loop is meant to be called in a while True loop
        '''
        if self.cur_episode < self.n_max_episodes:
            # Keep taking steps in this episode
            done = False
            if self.cur_step < self.n_max_steps:
                # Take another step in the environment
                self.obs, reward, done, self.next_value, self.next_action_dists, grads = self.play_one_step(self.obs, self.next_value, self.next_action_dists)
                self.episode_rewards.append(reward)
                self.episode_grads.append(grads)
                self.cur_step += 1
            if done or self.cur_step >= self.n_max_steps:
                # Done with this episode, reset environment for new episode
                self.all_rewards.append(self.episode_rewards)
                self.all_grads.append(self.episode_grads)
                self.next_value = None
                self.next_action_dists = None
                self.obs = self.env.reset()
                self.episode_rewards = []
                self.episode_grads = []
                self.cur_step = 0
                self.cur_episode += 1
        else:
            # Done iterating through all the episodes, apply gradients and reset for more training
            self.apply_grads(self.all_rewards, self.all_grads)
            self.cur_episode = 0
            self.all_rewards = []
            self.all_grads = []
