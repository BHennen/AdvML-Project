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
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
from multiprocessing import Queue
import pickle
import os

from queue import PriorityQueue

from car_racing_base import CarRacing
from reward_predictor import RewardPredictorModel
from manager import Message

PROCESS_ID = 'proc1'

class Agent():

    def __init__(self, env, lr=0.001):
        self.env = env
        self.learning_rate = lr

        self.policy = self._build_model()

    '''
    Intializes Keras model as policy
    '''
    def _build_model(self, do_dropout=True, p_dropout=0.5, l2_reg_val=0.001):
        state_shape = self.env.observation_space.shape
        input_tensor = Input(shape=state_shape, name='input')
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
        x = Dense(24, activation="relu", name='dense_1', kernel_regularizer=l2(l2_reg_val))(x)
        output1 = Dense(3, name='out1', activation='sigmoid')(x)
        output2 = Dense(2, name='out2', activation='sigmoid')(x)
        output3 = Dense(2, name='out3', activation='sigmoid')(x)
        model = Model(inputs=[input_tensor], outputs=[output1, output2, output3], name='agent_policy')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate, beta_1=0.99, epsilon=10**-5))
        print(model.summary())
        return model

    def select_action(self, observation):
        # Get action, append both to history
        predictions = self.policy.predict(np.array([observation]))
        action_idxs = np.array([np.argmax(pred_i) for pred_i in predictions])
        return action_idxs

class Segment():

    def __init__(self, trajectory, avg_variance):
        self.trajectory = trajectory
        self.length = len(self.trajectory)
        self.avg_variance = avg_variance


class SegmentSelector():
    SEGMENT_FILE = 'master.segs'
    SEGMENT_LENGTH = 100
    MIN_AVG_VARIANCE = 0.5
    RANDOM_POLICY_SEGMENT_COUNT = 50

    def __init__(self, trajectory_queue):
        self.trajectory_queue = trajectory_queue
        self.current_trajectory = list()
        self.current_variance = 0
        self.avg_variance = 0
        self.trajectory_pqueue = self.load_pqueue()
        self.total_frames = 0
        self.total_segments = 0

    def load_pqueue(self):
        if not os.path.exists(self.SEGMENT_FILE):
            return PriorityQueue()
        else:
            with open(self.SEGMENT_FILE, 'rb') as f_in:
                return pickle.load(f_in)

    def end_segment(self):
        self.trajectory_pqueue.put((self.avg_variance, Segment(self.current_trajectory, self.avg_variance)))
        self.current_trajectory = list()
        self.avg_variance = self.current_variance / len(self.current_trajectory)
        self.current_variance = 0
        self.total_segments += 1

        if self.total_segments >= self.RANDOM_POLICY_SEGMENT_COUNT:
            s1, s2 = self.trajectory_pqueue.get(), self.trajectory_pqueue.get()
            self.trajectory_queue.push(s1)

    def update_segment(self, trajectory, variance):
        obs, action = trajectory
        self.total_frames += 1
        self.current_trajectory.append(trajectory)
        self.current_variance += variance

        if len(trajectory) >= self.SEGMENT_LENGTH:
            self.end_segment()


def run_agent_process(traj_q):
    run_agent = True
    do_learn_policy = False
    render_game = True

    reward_predictor_model = RewardPredictorModel()

    # Setup Env
    env = CarRacing()

    # Setup Agent
    agent = Agent(env)

    # Setup segment selector
    seg_select = SegmentSelector(traj_q)

    current_state = env.reset()

    i_step = 0
    while run_agent:
        # Render (if desired)
        if render_game:
            env.render()

        # Get agent action
        action = agent.select_action(current_state)
        trajectory = (current_state, action)
        # Get state
        predicted_reward, variance = reward_predictor_model.predict([trajectory])

        # Update state (perform action)
        next_state, _, finished, _ = env.step(action)
        run_agent = not finished

        # Update segment
        seg_select.update_segment(trajectory, variance)

        # If we're training the policy, do so here
        if do_learn_policy:
            pass

        current_state = next_state

        # Update iterator
        i_step += 1

if __name__ == '__main__':
    run_agent_process(Queue())
