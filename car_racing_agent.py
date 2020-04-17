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
from multiprocessing import Queue, Pipe, current_process
from queue import Full, Empty

import heapq
import os
import itertools

import pyglet

from car_racing_base import CarRacing
from reward_predictor import RewardPredictorModel

PROCESS_ID = 'proc1'
CONSOLE_UPDATE_INTERVAL = 10

class Agent():
    # Discrete actions
    STEER_ACTIONS = [-1, 0, 1]
    GAS_ACTIONS = [0, 1]
    BRAKE_ACTIONS = [0, 0.8]

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

    def select_random_action(self, observation):
        steer = np.random.choice(self.STEER_ACTIONS)
        gas = np.random.choice(self.GAS_ACTIONS)
        brake = np.random.choice(self.BRAKE_ACTIONS)
        return [steer, gas, brake]


class SegmentQueue():
    def __init__(self, maxlen, verbose=False):
        self.verbose = verbose
        self.size = 0
        self.maxlen = maxlen
        self.counter = itertools.count()
        self.entry_finder = {} # Dictionary stores the actual trajectory
        self.pq_high = [] # Heap that produces segments with highest variance by heappop
        self.pq_low = [] # Heap that produces segments with lowest variance by heappop

    def is_full(self):
        return self.size >= self.maxlen

    def add_segment(self, trajectory, variance):
        ''' Adds a segment to the queue. Removes the lowest variance trajectory if full.
        '''
        traj_no = next(self.counter)
        if self.verbose: print(f"Adding segment {traj_no} to queue- variance={variance:.4f}")
        self.entry_finder[traj_no] = (trajectory, variance)

        heapq.heappush(self.pq_high, [-variance, traj_no])
        heapq.heappush(self.pq_low, [variance, traj_no])

        self.size += 1
        if self.size > self.maxlen:
            self.pop_lowest()

    def pop_highest(self):
        ''' Returns the segment with the highest variance
        '''
        while self.pq_high:
            variance, traj_no = heapq.heappop(self.pq_high)
            if traj_no in self.entry_finder:
                if self.verbose: print(f"Popping segment {traj_no} from queue- variance={np.abs(variance):.4f}")
                self.size -= 1
                return self.entry_finder.pop(traj_no)
        raise KeyError('pop from an empty segment queue')

    def pop_lowest(self):
        ''' Returns the segment with the lowest variance
        '''
        while self.pq_low:
            variance, traj_no = heapq.heappop(self.pq_low)
            if traj_no in self.entry_finder:
                if self.verbose: print(f"Popping segment {traj_no} from queue- variance={np.abs(variance):.4f}")
                self.size -= 1
                return self.entry_finder.pop(traj_no)
        raise KeyError('pop from an empty segment queue')


class SegmentSelector():
    SEGMENT_LENGTH = 5
    PQUEUE_MAX_SIZE = 5
    SAMPLE_SEGMENTS = 5 # Sample N new segments before sending a pair to the human

    def __init__(self, trajectory_queue, reward_predictor_model, verbose=False):
        self.trajectory_queue = trajectory_queue
        self.reward_predictor_model = reward_predictor_model
        self.current_trajectory = list()
        self.trajectory_pqueue = SegmentQueue(maxlen=self.PQUEUE_MAX_SIZE, verbose=verbose)
        self.sampled_segments = 0
        self.verbose = verbose

    def load_pqueue(self):
        ret = list()
        heapq.heapify(ret)
        return ret

    def end_segment(self):
        # Determine variance of trajectory
        reward, variance = self.reward_predictor_model.predict(self.current_trajectory)

        if self.verbose:
            print(f"<CarRacingAgent.SegmentSelector> - Recording trajectory, reward={reward},variance={variance}")

        # Put trajectory into segment queue, automatically removes the lowest variance trajectory
        self.trajectory_pqueue.add_segment(self.current_trajectory, variance)
        self.sampled_segments += 1

        # If the queue is full and we have processed enough segments, send pair to human
        if self.trajectory_pqueue.is_full() and self.sampled_segments >= self.SAMPLE_SEGMENTS:
            # Get two highest variance segments
            (s1, var1), (s2, var2) = self.trajectory_pqueue.pop_highest(), self.trajectory_pqueue.pop_highest()
            # send them to human
            try:
                if self.verbose:
                    print(f"<CarRacingAgent.SegmentSelector> - Sending trajectory pair to process 2, variances={var1:.4f}, {var2:.4f}")
                self.trajectory_queue.put_nowait((s1,s2))
            except Full as e:
                if self.verbose:
                    print(f"<CarRacingAgent.SegmentSelector> - Queue is full! Cannot send additional trajectories")
                self.trajectory_pqueue.add_segment(s1, var1)
                self.trajectory_pqueue.add_segment(s2, var2)
            self.sampled_segments = 0

        # Reset vars
        self.current_trajectory = list()

    def update_segment(self, trajectory, variance):
        self.current_trajectory.append(trajectory)

        if len(self.current_trajectory) >= self.SEGMENT_LENGTH:
            self.end_segment()


class AgentProcess(object):
    def __init__(self, traj_q, weight_q, mgr_pipe, verbose=False, render=False, profile=None):
        self.traj_q = traj_q
        self.weight_q = weight_q
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
        self.tot_pred_reward = 0
        self.tot_env_reward = 0
        self.profile = profile

    def _window_closed(self):
        print("closing car viewer")
        self.render_game=False        

    def _process_messages(self):
        # Check mgr pipe 
        while self.mgr_pipe.poll():
            msg = self.mgr_pipe.recv()
            # kill signal
            if msg.sender == "mgr" and msg.title == "stop":
                self._stop()
            # Render
            if msg.sender == "proc2" and msg.title == "render":
                self.render_game = not self.render_game
    
    def _update_reward_weights(self):
        # Check weight queue for new weights
        try:
            weights = self.weight_q.get_nowait()
            self.reward_predictor_model.set_weights(weights)
        except Empty:
            pass
        
    def _stop(self):
        ''' Called when process asked to terminate by mgr
        '''
        if self.has_rendered:
            self.env.viewer.window.pop_handlers()
            self.env.viewer.close()
        self.weight_q.close()
        self.mgr_pipe.close()
        pyglet.app.exit()        
        if self.profile:
            self.profile.disable()
            proc_name = ''.join(current_process().name.split())
            filename = os.path.join("profile", f"{proc_name}.profile")
            self.profile.dump_stats(filename)
        print(f"Quitting {current_process().name} process")
        os._exit(0)

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
        elif self.env.viewer.window.visible:
            self.env.viewer.window.set_visible(visible=False)

    def _reset(self):
        ''' Resets the training environment
        '''        
        # Set initial state
        if self.game > 0:
            print(f"Finished game {self.game}. Iterations: {self.i_step}. " + \
                  f"Final score: {self.tot_env_reward:.2f}. Predicted Score: {self.tot_pred_reward:.2f}")
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
        self._update_reward_weights()
        self._agent_step()
        self._render()

        self.i_step += 1

    def _run(self):
        ''' Main loop for process 1
        '''        
        pyglet.clock.schedule(self._main_loop) #Run program
        pyglet.app.run()
            

def run_agent_process(traj_q, weight_q, mgr_pipe, render=False, profile=False):
    prof = None
    if profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    
    agent_proc = AgentProcess(traj_q, weight_q, mgr_pipe, render=render, profile = prof)
    agent_proc._run()
    

if __name__ == '__main__':
    a, b = Pipe() # dummy pipe connections
    run_agent_process(Queue(), Queue(), b, render=True)
