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
from keras.initializers import RandomNormal
import numpy as np
from multiprocessing import Queue, Pipe
from queue import Full

import heapq

import pyglet

from car_racing_base import CarRacing
from reward_predictor import RewardPredictorModel

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
        x = Dense(24, activation="relu", name='dense_1', kernel_initializer=RandomNormal(), kernel_regularizer=l2(l2_reg_val))(x)
        output1 = Dense(3, name='out1', activation='softmax', kernel_initializer=RandomNormal())(x)
        output2 = Dense(2, name='out2', activation='softmax', kernel_initializer=RandomNormal())(x)
        output3 = Dense(2, name='out3', activation='softmax', kernel_initializer=RandomNormal())(x)
        model = Model(inputs=[input_tensor], outputs=[output1, output2, output3], name='agent_policy')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate, beta_1=0.99, epsilon=10**-5))
        if print_summary:
            model.summary()
        return model

    def select_action(self, observation):
        # Get action, append both to history
        predictions = self.policy.predict(np.array([observation]))
        predictions = [pred_i.flatten() for pred_i in predictions]
        action_idxs = []
        for pred_i in predictions:
            a = np.indices((len(pred_i),)).flatten()
            p = pred_i
            action_idxs.append(np.random.choice(a=a, size=1, p=p))
        action_idxs = np.array(action_idxs).flatten()
        return action_idxs

class Segment():
    def __init__(self, trajectory, variance):
        self.trajectory = trajectory
        self.variance = variance

    def __lt__(self, other):
        other: Segment
        return self.variance < other.variance


class SegmentSelector():
    SEGMENT_LENGTH = 5
    PQUEUE_MAX_SIZE = 5

    def __init__(self, trajectory_queue, reward_predictor_model, verbose=False):
        self.trajectory_queue = trajectory_queue
        self.reward_predictor_model = reward_predictor_model
        self.current_trajectory = list()
        self.trajectory_pqueue = list()
        heapq.heapify(self.trajectory_pqueue)
        self.total_frames = 0
        self.total_segments = 0
        self.initial_values_generated = False
        self.verbose = verbose

    def load_pqueue(self):
        ret = list()
        heapq.heapify(ret)
        return ret

    def end_segment(self):
        # Determine variance of trajectory
        reward, variance = self.reward_predictor_model.predict(self.current_trajectory)
        ins_trajectory = Segment(self.current_trajectory, -variance)

        if self.verbose:
            print(f"<CarRacingAgent.SegmentSelector> - Recording trajectory, reward={reward},variance={variance}")

        # Put trajectory into heap
        if self.total_segments < self.PQUEUE_MAX_SIZE:
            self.total_segments += 1
            heapq.heappush(self.trajectory_pqueue, ins_trajectory)
        elif self.total_segments > 1:
            heapq.heapreplace(self.trajectory_pqueue, ins_trajectory)
            pair = self.get_next_segments()
            try:
                if self.verbose:
                    print(f"<CarRacingAgent.SegmentSelector> - Sending trajectory to process 3, variance={-variance}")
                self.trajectory_queue.put_nowait(pair)
                self.total_segments -= 2
            except Full as e:
                if self.verbose:
                    print(f"<CarRacingAgent.SegmentSelector> - Queue is full! Cannot send additional trajectories")
                s1, s2 = pair
                heapq.heappush(self.trajectory_pqueue, s1)
                heapq.heappush(self.trajectory_pqueue, s2)
        # Reset vars
        self.current_trajectory = list()

    def get_next_segments(self):
        s1, s2 = heapq.heappop(self.trajectory_pqueue), heapq.heappop(self.trajectory_pqueue)
        return (s1.trajectory, s2.trajectory)

    def update_segment(self, trajectory, variance):
        self.total_frames += 1
        self.current_trajectory.append(trajectory)

        if len(self.current_trajectory) >= self.SEGMENT_LENGTH:
            self.end_segment()

        if self.initial_values_generated:
            try:
                self.trajectory_queue.put_nowait(self.get_next_segments())
            except Full as e:
                pass

class AgentProcess(object):
    def __init__(self, traj_q, weight_pipe, mgr_pipe, verbose=False, render=False):
        self.traj_q = traj_q
        self.weight_pipe = weight_pipe
        self.mgr_pipe = mgr_pipe
        self.mgr_kill_sig = False
        self.reward_predictor_model = RewardPredictorModel()
        self.seg_select = SegmentSelector(traj_q, self.reward_predictor_model, verbose=verbose)
        self.reset = True
        self.do_learn_policy = False
        self.render_game = render
        self.has_rendered = False
        self.env = CarRacing()
        self.agent = Agent(self.env)
        self.current_state = None
        self.verbose = verbose
        self.game = 0
        self.i_step = 0

    def _window_closed(self):
        print("closing car viewer")
        self.render_game=False
        self.env.viewer.window.set_visible(visible=False)

    def _process_messages(self):
        # Check weight pipe for new weights
        msg = None
        while self.weight_pipe.poll():
            msg = self.weight_pipe.recv()
        if msg and msg.sender == "proc3" and msg.title == "weights":
            self.reward_predictor_model.set_weights(msg.content)

        # Check mgr pipe 
        while self.mgr_pipe.poll():
            msg = self.mgr_pipe.recv()
            # kill signal
            if msg.sender == "mgr" and msg.title == "stop":
                self.mgr_kill_sig = True
            # Render
            if msg.sender == "proc2" and msg.title == "render":
                self.render_game = True

    def _stop(self):
        ''' Called when process asked to terminate by mgr
        '''
        if self.has_rendered:
            self.env.viewer.window.pop_handlers()
            self.env.viewer.close()
        pyglet.app.exit()

    def _render(self):
        ''' Render (if desired)
        '''
        if self.render_game:
            if not self.env.viewer.window.visible:
                self.env.viewer.window.set_visible()
            if not self.has_rendered:
                #one time initialization
                self.has_rendered = True
                self.env.viewer.window.push_handlers(on_close = self._window_closed)
            else:
                self.env.render()

    def _reset(self):
        ''' Resets the training environment
        '''        
        # Set initial state
        if self.game > 0:
            print(f"Finished game {self.game}. Iterations: {self.i_step}. Final score: {self.tot_env_reward}. Predicted Score: {self.tot_pred_reward}")
        self.game += 1
        self.current_state = self.env.reset()
        self.i_step = 0
        self.tot_pred_reward = 0
        self.tot_env_reward = 0

    def _agent_step(self):
        ''' One step of the agent
        '''
        if self.reset:
            self._reset()

        # Get agent action
        action = self.agent.select_action(self.current_state)
        trajectory = (self.current_state, action)
        # Get state
        predicted_reward, variance = self.reward_predictor_model.predict([trajectory])
        self.tot_pred_reward += predicted_reward

        # Update state (perform action)
        next_state, reward, finished, _ = self.env.step(action)
        self.reset = finished
        self.tot_env_reward += reward

        # Update segment
        self.seg_select.update_segment(trajectory, variance)

        # If we're training the policy, do so here
        if self.do_learn_policy:
            pass

        self.current_state = next_state

    def _main_loop(self, dt=None):
        # Debug output
        if self.i_step % CONSOLE_UPDATE_INTERVAL == 0:
            print(f"Update: Iteration={self.i_step}")

        self._process_messages()
        if self.mgr_kill_sig:
            self._stop()
            return False # Break out of loop to end process

        self._agent_step()
        self._render()

        self.i_step += 1

        return True

    def run(self):
        ''' Main loop for process 1
        '''

        # Main process loop
        pyglet.clock.schedule(self._main_loop) #Run program
        pyglet.app.run()
        print("Quitting process 1")
            

def run_agent_process(traj_q, weight_pipe, mgr_pipe, render=False):
    agent_proc = AgentProcess(traj_q, weight_pipe, mgr_pipe, render=render)
    agent_proc.run()

if __name__ == '__main__':
    a, b = Pipe() # dummy pipe connections
    run_agent_process(Queue(), a, b, render=True)
