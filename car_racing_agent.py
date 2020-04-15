# A trajectory segment is a sequence of observations and actions, σ = ((o0,a0),(o1,a1),...,(ok−1,ak−1))∈(O×A)k. 
# 
# These networks are updated by three processes:
# 1. The policy π interacts with the environment to produce a set of trajectories {τ1,...,τi}. The parameters of π 
#    are updated by a traditional reinforcement learning algorithm, in order to maximize the sum of the predicted
#    rewards rt = r(ot, at).
# 2. We select pairs of segments (σ1,σ2) from the trajectories {τ1,...,τi} produced in step 1, and send them to a
#    human for comparison.
# 3. The parameters of the mapping r are optimized via supervised learning to fit the comparisons collected from
#    the human so far.

# This script is for process 1:
# 1) Use policy gradient RL algorithm to optimize sum of predicted rewards
# 2) Capture pairs of trajectory segments to use for process 2. Sample a large number of pairs of trajectory segments
#    of length k, use each reward predictor in our ensemble (from process 3) to predict which segment will be preferred
#    from each pair, and then select those trajectories for which the predictions have the highest variance across ensemble members.
# 3) Send chosen pair of trajectory segment to process 2
from keras import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from car_racing_base import CarRacing
import numpy as np

from multiprocessing import Queue

class Agent():

    def __init__(self, env, lr=0.001):
        self.env = env
        self.learning_rate = lr

        self.policy = self.init_policy()

    '''
    Intializes Keras model as policy
    '''
    def init_policy(self, do_dropout=True, p_dropout=0.5, l2_reg_val=0.001):
        state_shape = self.env.observation_space.shape
        input_tensor = InputLayer(input_shape=state_shape[0])
        if do_dropout:
            input_tensor = input_tensor(Dropout(p_dropout))
        # Convolution layers
        conv1 = Conv2D(15, kernel_size=(7, 7), strides=3, name='conv1')(input_tensor)
        conv2 = Conv2D(15, kernel_size=(5, 5), strides=2, name='conv2')(conv1)
        conv3 = Conv2D(15, kernel_size=(3, 3), strides=1, name='conv3')(conv2)
        conv4 = Conv2D(15, kernel_size=(3, 3), strides=1, name='conv4')(conv3)
        maxpool1 = MaxPooling2D((2, 2), name='pool_1')(conv4)
        flatten1 = Flatten(name='flatten')(maxpool1)
        # Dense layers
        dense1 = Dense(24, activation="relu", name='dense_1', kernel_regularizer=l2(l2_reg_val))(flatten1)
        output1 = Dense(3, name='out1', activation='sigmoid')(dense1)
        output2 = Dense(2, name='out2', activation='sigmoid')(dense1)
        output3 = Dense(2, name='out3', activation='sigmoid')(dense1)
        model = Model(inputs=[input_tensor], outputs=[output1, output2, output3])
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate, beta_1=0.99, epsilon=10**-5))
        return model

    def get_next_action(self, observation):
        # Get action, append both to history
        return np.argmax(self.policy.predict(observation))

class Segment():

    def __init__(self, trajectory, avg_variance):
        self.trajectory = trajectory
        self.length = len(self.trajectory)
        self.avg_variance = avg_variance


class SegmentSelector():
    SEGMENT_LENGTH = 100

    def __init__(self, trajectory_queue):
        self.trajectory_queue = trajectory_queue
        self.current_trajectory = list()
        self.current_variance = 0
        self.avg_variance = 0

    def end_trajectory(self):
        self.current_trajectory = list()
        self.avg_variance = self.current_variance / len(self.current_trajectory)
        self.current_variance = 0

    def update_trajectory(self, trajectory):
        obs, action, reward_variance = trajectory
        self.current_trajectory.append(trajectory)
        self.current_variance += reward_variance

        if len(trajectory) >= self.SEGMENT_LENGTH:
            self.end_trajectory()


def run_agent_process(traj_q):
    run_agent = True
    render_game = True

    # Setup Env
    env = CarRacing()

    # Setup Agent
    agent = Agent(env, traj_q)

    current_state = env.reset()

    while run_agent:
        # Render (if desired)
        if render_game:
            env.render()
        # Get agent action
        action = agent.get_next_action(current_state)
        # Update state
        next_state, _, finished = env.step(action)
        run_agent = not finished


if __name__ == '__main__':
    run_agent_process(Queue())
