from queue import Full, Empty

from collections import deque

# from keras.layers import Input, Conv2D, Dense
# from keras.models import Model

# Overall processes:
# A trajectory segment is a sequence of observations and actions, σ = ((o0,a0),(o1,a1),...,(ok−1,ak−1))∈(O×A)k. 
# Write σ1 > σ2 to indicate that the human preferred trajectory segment σ1 to trajectory segment σ2.
#
# These networks are updated by three processes:
# 1. The policy π interacts with the environment to produce a set of trajectories {τ1,...,τi}. The parameters of π 
#    are updated by a traditional reinforcement learning algorithm, in order to maximize the sum of the predicted
#    rewards rt = r(ot, at).
# 2. We select pairs of segments (σ1,σ2) from the trajectories {τ1,...,τi} produced in step 1, and send them to a
#    human for comparison.
# 3. The parameters of the mapping r are optimized via supervised learning to fit the comparisons collected from
#    the human so far.

# This script is for process 3:
# 1) Receive evaluated trajectory tuples (σ1, σ2, u) from process 2 into a queue
# 2) Agent retrieves item from queue and stores them locally into a circular buffer to hold the last N comparisons
# 3) Triple (σ1, σ2, μ) is evaluated by agent, improving neural net model that predicts reward.
# 4) Parameters for the model are saved into a variable that can be accessed by process 1.
#
#

BUFFER_LEN = 3000

class RewardPredictor(object):
    '''
    '''
    def __init__(self, pref_q, weight_conn, mgr_conn, buffer_len):
        self._pref_q = pref_q
        self._weight_conn = weight_conn
        self._mgr_conn = mgr_conn
        self._q = deque(maxlen = buffer_len)
        self._init_model()
        self._stop_sig = False
        
    def _get_comparisons(self):
        # Gets all available comparisons and stores them in queue
        while True:
            try:
                triple = self._pref_q.get_nowait()
                self._q.append(triple)
            except Empty:
                break
    
    def _output_model_weights(self):
        # TODO: Outputs the current model weights to the weight connection
        pass

    def _init_model(self):
        # TODO: Create a compiled and working model to be used for learning.
        self.model = self.build_model()
    
    def _learn(self):
        # TODO: Learns from the buffer
        pass

    def _check_msgs(self):
        # Read manager signals
        msgs = []
        while self._mgr_conn.poll():
            msgs.append(self._mgr_conn.recv())

        for msg in msgs:
            if msg == "stop":
                self.stop()

    def stop(self):
        # Stop process execution
        print("Quitting reward predictor")
        self._stop_sig = True

    def build_model(self):
        # TODO: Builds and return a model 
        return None

    def run(self):
        # Main process loop
        while True:
            self._check_msgs()
            if self._stop_sig:
                # Program signalled to stop, end loop
                break
            self._get_comparisons()
            self._learn()
            self._output_model_weights()


def run_reward_predictor(pref_q, weight_conn, mgr_conn):
    reward_predictor = RewardPredictor(pref_q=pref_q, weight_conn=weight_conn, mgr_conn=mgr_conn, buffer_len=BUFFER_LEN)
    reward_predictor.run()

if __name__ == "__main__":
    #TODO: test code
    pass