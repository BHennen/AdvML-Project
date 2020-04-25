import tensorflow as tf
import tensorflow.keras as keras
from keras.regularizers import l2
from keras.initializers import RandomNormal
from tensorflow.keras.layers import LeakyReLU, UpSampling1D, Input, InputLayer, Reshape, Activation, Lambda, AveragePooling1D
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

import numpy as np

import os, sys
import copy

def discount_vector(x, discount, last_val=0):
    ''' Returns discounted array of x
    '''
    discounted = np.array(x, dtype=np.float32)
    discounted[-1] += last_val
    for step in range(discounted.size - 2, -1 , -1):
        discounted[step] += discounted[step + 1] * discount
    return discounted

class Memory:
    """ Based on: https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, size, obs_dim, num_actions, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32) # Observations
        self.act_buf = np.zeros((size, num_actions), dtype=np.int32) # Action indices chosen
        self.rew_buf = np.zeros(size, dtype=np.float32) # Rewards
        self.adv_buf = np.zeros((size, 1), dtype=np.float32) # Advantages
        self.ret_buf = np.zeros((size, 1), dtype=np.float32) # Returns
        self.val_buf = np.zeros(size, dtype=np.float32) # Estimated values
        self.probbuf = np.zeros((size, num_actions), dtype=np.float32) # Probability of choosing action
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, prob):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.probbuf[self.ptr] = prob
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_vector(deltas, self.gamma * self.lam).reshape((deltas.size, 1))
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_vector(rews, self.gamma).reshape((rews.size, 1))[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + + 1e-8)
        return copy.deepcopy([self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.probbuf])

    def copy_params(self):
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        adv_buf = (self.adv_buf - adv_mean) / (adv_std + + 1e-8)
        return copy.deepcopy([self.obs_buf, self.act_buf, adv_buf, self.ret_buf, self.probbuf])
    
    

class ActorCriticTrainer(object):
    ''' Advantage Actor Critic that is updated in batches. Needs model supplied to it.

    '''
    def __init__(self, env, action_values, obs_dim, model, optimizer, steps_per_episode=200, steps_per_epoch=1000, 
                 clip_ratio=0.2, gamma=0.99, beta=0.01, delta=0.5, lam=0.95, frameskip=1, reward_fn=None, verbosity=0):
        '''
        :param frameskip: Allows the agent to skip n-1 simulations, repeating the chosen action for n frames.
        :param action_values: 2D list of values to send to the environment depending on which action is chosen. Should
                              be a list of actions which contain a list of possible values, eg: [[1,2,3],[5,6]]
        
        :param gamma: Controls the discount rate of rewards.        
        :param beta: Controls the contribution of the entropy term for loss.
        :param delta: Controls the contribution of the critic for loss.
        :param steps_per_episode: Number of steps before the episode is terminated (or done, whichever comes first)
        :param steps_per_epoch: Number of steps before gradient is updated.
        '''
        self.model = model
        self.env = env
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.optimizer = optimizer
        self.action_values = action_values
        self.steps_per_episode = steps_per_episode
        self.steps_per_epoch = steps_per_epoch
        self.action_indices = [[i for i in range(len(action_vals))] for action_vals in action_values]
        self.num_unique_actions = len(np.unique(self.action_indices))
        self.reward_fn = reward_fn
        self.frameskip = frameskip
        self.cur_episode = 0
        self.cur_step = 0
        self.ep_step = 0
        self.obs = self.env.reset()
        self.verbosity = verbosity
        self.mem = Memory(steps_per_epoch, obs_dim, len(action_values), gamma, lam)
        self.prev_mem = None

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def play_one_epoch(self, step_done_fn = None, ep_done_fn = None, epoch_done_fn = None, training = False):
        # Take another step in the environment
        self.obs, reward, done = self._play_one_step(self.obs, step_done_fn, training)            
        self.cur_step += 1
        self.ep_step += 1
        if step_done_fn: step_done_fn(ep_step = self.ep_step)

        # Check if done with episode or cutoff by epoch
        if done or self.ep_step >= self.steps_per_episode or self.cur_step >= self.steps_per_epoch:
            # Done with this episode, or epoch cutoff
            if training:
                last_val = 0 if done else self.model(inputs = tf.convert_to_tensor([self.obs]))[0] #TODO: check if correct, should be one value
                self.mem.finish_path(last_val)
            if ep_done_fn:
                ep_done_fn(ep_step=self.ep_step, terminal = done or self.ep_step >= self.steps_per_episode)
            self.ep_step = 0
            self.obs = self.env.reset()

        # Check if done with epoch
        if self.cur_step >= self.steps_per_epoch:
            self.cur_step  = 0
            losses = None
            if training:
                losses = self._apply_grads(done)
            if epoch_done_fn: epoch_done_fn(losses = losses)

    def _play_one_step(self, obs, step_done_fn = None, training = False):
        ''' Plays one step of the environment
        '''

        # Get the estimated value and probability distributions for the actions
        value, *logits = self.model(inputs = tf.convert_to_tensor([obs]))
        probs = [tf.nn.softmax(logit).numpy()[0] for logit in logits]

        # Select an action index from the distributions for all actions
        if training:
            # Training selects actions based on the probabilities
            action_indices = [np.random.choice(self.action_indices[which_action], p=prob)
                              for which_action, prob in enumerate(probs)]
        else:
            # Not training selects the best (most favored) action
            action_indices = [np.argmax(prob) for prob in probs]
                          
        # Select the actual actions to give to the environment
        actions = [self.action_values[which_action][action_idx]
                   for which_action, action_idx in enumerate(action_indices)]
        if len(actions) == 1:
            actions = actions[0]

        # and take n steps in the environment
        for _ in range(self.frameskip):
            next_obs, reward, done, _ = self.env.step(actions)
            if done:
                reward = -1
                break

        # Store this step in the memory 
        if training:
            act_probs = [prob[action_indices[which_action]] for which_action, prob in enumerate(probs)]
            self.mem.store(obs, action_indices, reward, value, act_probs)       
         
        return next_obs, reward, done
    
    def _apply_grads(self, done):
        '''
        Calculates loss with the current memory and applies gradients to the model.
        '''
        # Get the buffer of previous states
        cur_mem = self.mem.get()
        if self.prev_mem is None:
            self.prev_mem = cur_mem # First time running there is no previous policy

        prev_obs_buf, prev_act_buf, prev_adv_buf, prev_ret_buf, prev_probbuf = self.prev_mem

        # Record the loss and update gradients
        with tf.GradientTape() as tape:
            losses = self._calc_loss_ppo(prev_obs_buf, prev_act_buf, prev_adv_buf, prev_ret_buf, prev_probbuf)
        
        grads = tape.gradient(losses[0], self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        # Done updating policy, associate current mem with prev policy
        self.prev_mem = cur_mem
        return losses

    def _calc_loss_ppo(self, obs_buf, act_buf, adv_buf, ret_buf, probbuf):
        # Calculate policy loss using previous policy information
        # Get the estimated value and probability distributions for the previous states using the new policy
        values, *logits = self.model(inputs = tf.convert_to_tensor(obs_buf, dtype=tf.float32))
        # Get action probability distributions
        dists = [tf.nn.softmax(logit) for logit in logits]
        # From the old state's action indices get the probability of choosing that action using new policy
        new_act_proba = tf.convert_to_tensor([[dists[which_action][state][action_ind]
                                              for which_action, action_ind in enumerate(actions)]
                                              for state, actions in enumerate(act_buf)])

        ratio = tf.math.divide_no_nan(new_act_proba, probbuf)          # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_buf>=0, (1+self.clip_ratio)*adv_buf, (1-self.clip_ratio)*adv_buf)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv_buf, min_adv))

        # Calculate current value loss and entropy loss using current observations
        # get log probabilities of the distributions
        log_dists = [tf.math.log(dist + 1e-20) for dist in dists]
        value_loss = self.delta * tf.reduce_mean((ret_buf - values)**2)
        entropy_loss = self.beta * tf.reduce_mean(
                                        tf.reduce_sum(
                                            [tf.reduce_sum(tf.math.multiply_no_nan(log_dist, dist), axis=-1)
                                                for dist, log_dist in zip(dists, log_dists)], axis=0))
        total_loss = policy_loss + value_loss + entropy_loss
        return total_loss, policy_loss, value_loss, entropy_loss


def build_cartpole_model(input_shape, action_values):
    # Simple model for testing
    inputs = Input(shape=input_shape, name='input')

    # Policy
    p_d1 = Dense(100, activation='relu', name='policy_dense_1')(inputs)
    policy_logits = []
    for action in action_values:
        policy_logits.append(Dense(len(action))(p_d1))
    
    # Values
    v_d1 = Dense(100, activation='relu', name='values_dense_1')(inputs)
    values = Dense(1)(v_d1)

    model = Model(inputs = [inputs], outputs=[values] + policy_logits, name="Actor-Critic Cartpole")
    return model

def train(ac_trainer, path):
    render_flag = False
    def step_done_fn(ep_step, *args, **kwargs):
        nonlocal render_flag
        if render_flag: env.render()

    consecutive = 0
    tot_steps = []
    num_episodes = 0
    training_done = False
    def ep_done_fn(ep_step, terminal, *args, **kwargs):
        nonlocal render_flag, consecutive, tot_steps, num_episodes, training_done
        if terminal: # Episode reached max steps or died; wasn't cut off by epoch
            tot_steps.append(ep_step)
            num_episodes += 1
            if num_episodes % 20 == 0:
                avg_steps = np.mean(tot_steps[-20:])
                print(f"Episode {num_episodes} done. Steps: {ep_step}. Last 20 mean steps: {avg_steps}.")

        complete = ep_step >= 200 
        if complete or num_episodes % 50 == 0:
            render_flag = True
        else:
            render_flag = False
        if complete:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive > 100:
            print(f"Game solved in {num_episodes} episodes, and {np.sum(tot_steps)} steps.")
            training_done = True

    tot_losses = []
    def epoch_done_fn(losses, *args, **kwargs):
        total_loss, policy_loss, value_loss, entropy_loss = losses
        tot_losses.append(total_loss)
        print(f"Epoch done. Last 10 avg_tot_loss:{np.mean(tot_losses[-10:]):.4f}, " +
              f"Tot_loss:{total_loss:.4f}, " +
              f"Pi_loss:{policy_loss:.4f}, " +
              f"Val_loss:{value_loss:.4f}, " +
              f"Ent_loss:{entropy_loss:.4f}"
                )

    while not training_done:
        ac_trainer.play_one_epoch(step_done_fn = step_done_fn,
                                  ep_done_fn = ep_done_fn,
                                  epoch_done_fn = epoch_done_fn,
                                  training = True)
    
    # training done, save file
    ac_trainer.save_weights(path)

def play(ac_trainer, path):
    ac_trainer.load_weights(path)

    def step_done_fn(*args, **kwargs):
        ac_trainer.env.render()

    def ep_done_fn(ep_step, *args, **kwargs):
        print(f"Episode done. Steps: {ep_step}.")

    try:
      while True:
        ac_trainer.play_one_epoch(step_done_fn = step_done_fn,
                                  ep_done_fn = ep_done_fn,
                                  epoch_done_fn = None,
                                  training = False)
    except KeyboardInterrupt:
      print("Received Keyboard Interrupt. Shutting down.")
    finally:
      ac_trainer.env.close()


if __name__ == "__main__":
    import gym, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Train the test program.")
    parser.add_argument("--path", default="cartpole_test.h5", help="Path to save or load the trained model weights.")
    args = parser.parse_args()
    env = gym.make("CartPole-v1")
    action_values = [[0, 1]]
    
    ac_model = build_cartpole_model(input_shape = (env.observation_space.shape[0],), action_values = action_values)
    optimizer = keras.optimizers.Adam(learning_rate = 1e-3)

    if args.train:
        steps_per_epoch = 200
    else:
        steps_per_epoch = 200

    ac_trainer = ActorCriticTrainer(env = env,
                                    action_values = action_values,
                                    obs_dim = (env.observation_space.shape[0],),
                                    model = ac_model,
                                    optimizer = optimizer,
                                    steps_per_episode = 200,
                                    steps_per_epoch = steps_per_epoch,
                                    clip_ratio = 0.2,
                                    gamma = 0.99,
                                    beta = 0.01,
                                    delta = 0.5,
                                    lam = 0.95,
                                    frameskip = 1,
                                    reward_fn = None,
                                    verbosity = 0
                                    )
                                     
    if args.train:
        train(ac_trainer, args.path)
    else:
        play(ac_trainer, args.path)

    
    