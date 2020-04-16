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
from multiprocessing import Queue, Pipe
from queue import Full

import heapq

from car_racing_base import CarRacing
from reward_predictor import RewardPredictorModel
from communication import Message

PROCESS_ID = 'proc1'
CONSOLE_UPDATE_INTERVAL = 10

class Agent():

    def __init__(self, env, lr=0.001):
        self.env = env
        self.learning_rate = lr

        self.policy = self._build_model()

    '''
    Intializes Keras model as policy
    '''
    def _build_model(self, do_dropout=True, p_dropout=0.5, l2_reg_val=0.001, print_summary=False):
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
        if print_summary:
            model.summary()
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
    SEGMENT_LENGTH = 100
    MIN_AVG_VARIANCE = 0.5
    PQUEUE_MAX_SIZE = 50
    MAX_TQUEUE_SIZE = 50

    def __init__(self, trajectory_queue):
        self.trajectory_queue = trajectory_queue
        self.current_trajectory = list()
        self.current_variance = 0
        self.avg_variance = 0
        self.trajectory_pqueue = list()
        heapq.heapify(self.trajectory_pqueue)
        self.total_frames = 0
        self.total_segments = 0
        self.initial_values_generated = False

    def load_pqueue(self):
        ret = list()
        heapq.heapify(ret)
        return ret

    def end_segment(self):
        heapq.heappush(self.trajectory_pqueue, (-self.avg_variance, Segment(self.current_trajectory, self.avg_variance)))
        self.current_trajectory = list()
        self.avg_variance = self.current_variance / len(self.current_trajectory)
        self.current_variance = 0
        self.total_segments += 1

        if self.total_segments >= self.PQUEUE_MAX_SIZE:
            self.initial_values_generated = True
            msg = self.get_segments()
            try:
                self.trajectory_queue.put_nowait(msg)
            except Full as e:
                pass

    def get_segments(self):
        s1, s2 = heapq.heappop(self.trajectory_pqueue), heapq.heappop(self.trajectory_pqueue)
        var, traj = s1
        var = -var
        s1 = (var, traj)
        var, traj = s2
        var = -var
        s2 = (var, traj)
        msg = (s1, s2)
        return msg

    def update_segment(self, trajectory, variance):
        self.total_frames += 1
        self.current_trajectory.append(trajectory)
        self.current_variance += variance

        if len(trajectory) >= self.SEGMENT_LENGTH:
            self.end_segment()

        if self.initial_values_generated:
            try:
                self.trajectory_queue.put_nowait(self.get_segments())
            except Full as e:
                pass

class AgentProcess(object):
    def __init__(self, traj_q, weight_pipe, mgr_pipe):
        self.traj_q = traj_q
        self.weight_pipe = weight_pipe
        self.mgr_pipe = mgr_pipe
        self.mgr_kill_sig = False
        self.reward_predictor_model = RewardPredictorModel()
        self.seg_select = SegmentSelector(traj_q)        
        self.run_agent = True
        self.do_learn_policy = False
        self.render_game = True
        self.env = CarRacing()
        self.agent = Agent(self.env)

    def _process_messages(self):
        # Check weight pipe for new weights
        msg = None
        while self.weight_pipe.poll():
            msg = self.weight_pipe.recv()
        if msg and msg.sender == "proc3" and msg.title == "weights":
            self.reward_predictor_model.set_weights(msg.content)

        # Check mgr pipe for kill signal
        while self.mgr_pipe.poll():
            msg = self.mgr_pipe.recv()
            if msg.sender == "mgr" and msg.title == "stop":
                self.mgr_kill_sig = True

    def _stop(self):
        ''' Called when process asked to terminate by mgr
        '''
        pass

    def _agent_step(self):
        ''' One step of the agent
        '''
        # Render (if desired)
        if self.render_game:
            self.env.render()

        # Get agent action
        action = self.agent.select_action(self.current_state)
        trajectory = (current_state, action)
        # Get state
        predicted_reward, variance = self.reward_predictor_model.predict([trajectory])

        # Update state (perform action)
        next_state, _, finished, _ = self.env.step(action)
        self.run_agent = not finished

        # Update segment
        self.seg_select.update_segment(trajectory, variance)

        # If we're training the policy, do so here
        if self.do_learn_policy:
            pass

        self.current_state = next_state

    def run(self):
        ''' Main loop for process 3
        '''
        # Set initial state
        self.current_state = env.reset()

        # Main process loop
        i_step = 0
        while True:
            # Debug output
            if i_step % CONSOLE_UPDATE_INTERVAL == 0:
                print(f"Update: Iteration={i_step}")

            self._process_messages()
            if self.mgr_kill_sig:
                self._stop()
                break # Break out of loop to end process

            self._agent_step()

            i_step += 1


def run_agent_process(traj_q, weight_pipe, mgr_pipe):
    agent_proc = AgentProcess(traj_q, weight_pipe, mgr_pipe)
    agent_proc.run()

if __name__ == '__main__':
    a, b = Pipe() # dummy pipe connections
    run_agent_process(Queue(), a, b)
