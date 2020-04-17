from communication import Message

from multiprocessing import current_process
import os
from queue import Full, Empty
from collections import deque

import numpy as np

# tensorflow, keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU, UpSampling1D, Input, InputLayer, Reshape, Activation, Lambda, AveragePooling1D
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Sequential, Model


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

class RewardPredictorModel(object):
    '''
    Model to be trained using user preferences of (trajectory, )
    '''
    def __init__(self, ensemble_size=3, l_rate=0.001):
        self.models=[]
        for _ in range(ensemble_size):
            self.models.append(self._build_model())
        self.optimizer = keras.optimizers.Adam(learning_rate=l_rate)

    def fit(self, triple):
        '''Fits a triple of (trajectory_1, trajectory_2, preference) to the model, updating the model weights
        '''
        traj_1, traj_2, pref = triple
        obs_1, actions_1 = list(zip(*traj_1))
        obs_1, actions_1 = np.array(obs_1, dtype=np.float32), np.array(actions_1)
        obs_2, actions_2 = list(zip(*traj_2))
        obs_2, actions_2 = np.array(obs_2, dtype=np.float32), np.array(actions_2)
        for model in self.models: # type: Model
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer. The operations that the layer applies
                # to its inputs are going to be recorded on the GradientTape.
                # Predictions for this trajectory pair
                t1_r_hat = model(inputs={"obs_input":obs_1, "act_input":actions_1}, training=True)
                t2_r_hat = model(inputs={"obs_input":obs_2, "act_input":actions_2}, training=True)
                # Compute the loss value for this triple.
                main_loss = self._loss_fn(t1_r_hat, t2_r_hat, pref)
                model_loss = model.losses
                loss_value = tf.add_n([main_loss] + model_loss)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_variables)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def predict(self, trajectory):
        '''
        Predicts reward using the given trajectory.
        Trajectory should be a list of tuples [(observation_1, action_1)...]
        Returns a tuple of (reward, variance)
        '''
        predictions = []
        obs, acts = list(zip(*trajectory))
        obs, acts = np.array(obs), np.array(acts)
        for model in self.models: # type: Model
            rewards = model.predict({"obs_input": obs, "act_input": acts})
            total_reward = np.sum(rewards)
            predictions.append(total_reward)
        avg_prediction = np.mean(predictions)
        variance = np.var(predictions)
        return (avg_prediction, variance)
    
    def get_weights(self):
        # Return a list of weights associated with this model ensemble
        weights = []
        for model in self.models:
            weights.append(model.get_weights())
        return weights
    
    def set_weights(self, weights):
        # Sets the weights of models in the ensemble. Should be compatible with get_weights
        for index, weight in enumerate(weights):
            self.models[index].set_weights(weight)

    # Loss function as described in paper by Christiano, Paul et. al. 2017
    def _loss_fn(self, t1_r_hat, t2_r_hat, pref):
        # Compute terms for softmax
        t1_rsum = tf.math.reduce_sum(t1_r_hat)
        t2_rsum = tf.math.reduce_sum(t2_r_hat)
        max_reward = tf.math.maximum(t1_rsum, t2_rsum)
        
        # Subtract max_reward to prevent overflow
        t1_exp = tf.math.exp(t1_rsum - max_reward)
        t2_exp = tf.math.exp(t2_rsum - max_reward)

        p_hat_1_gt_2 = t1_exp / (t1_exp + t2_exp)
        p_hat_2_gt_1 = t2_exp / (t2_exp + t1_exp)

        # Adjust probabilities that they prefer one over the other
        # by assuming they choose correctly 90% of the time, and
        # incorrectly 10% of the time
        p_hat_1_gt_2_adj = p_hat_1_gt_2 * 0.9 + p_hat_2_gt_1 * 0.1
        p_hat_2_gt_1_adj = p_hat_2_gt_1 * 0.9 + p_hat_1_gt_2 * 0.1

        log_phat1 = tf.math.log(p_hat_1_gt_2_adj)
        log_phat2 = tf.math.log(p_hat_2_gt_1_adj)

        loss = -1 * (pref[0] * log_phat1 + pref[1] * log_phat2)

        return loss

    def _build_cnn(self, width=96, height=96, depth=3, print_summary=False):
        # Build convolutional model for images input using similar architecture as Christiano 2017
        # TODO: use L2 regularization with the adapative scheme in Section 2.2.3
        obs_input = Input(shape=(height, width, depth), name="obs_input")
        fks = [(8, (7,7), 3), 
               (16, (5,5), 2),
               (32, (3,3), 2),
               (64, (3,3), 1)]
        x = obs_input
        for index, (filters, kernel, strides) in enumerate(fks):
            x = Conv2D(filters = filters, kernel_size = kernel, strides = strides, padding='same', name=f"cnn_conv{index+1}")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=0.5)(x)
        
        x = Flatten()(x)
        x = Dense(16, activation="elu", name = "cnn_dense1")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(4, activation="elu", name = "cnn_dense2")(x)

        model = Model(inputs=obs_input, outputs=x, name="CNN")
        if print_summary: model.summary()
        # return the CNN
        return model

    def _build_nn(self, print_summary=False):
        action_input = Input(shape = (3,), name="act_input") # gas, brake, steer action.
        x = Dense(16, activation="elu", name = "nn_dense1")(action_input)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(4, activation="elu", name = "nn_dense2")(x)
        model = Model(inputs=action_input, outputs=x, name="NN")
        if print_summary: model.summary()
        # return the NN
        return model

    def _build_model(self, print_summary=False):
        # Builds and return a model with both inputs
        cnn = self._build_cnn()
        nn = self._build_nn()
        combined = concatenate([cnn.output, nn.output])
        x = Dense(16, activation="relu", name="comb_dense1")(combined)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(1, activation="linear", name="output")(x) # linear prediction of the reward
        # normalize the rewards produced by rˆ to have mean close to 0 and standard deviation close to 1.
        x = BatchNormalization()(x) 
        model = Model(inputs=[cnn.input, nn.input], outputs=x)
        if print_summary:
            model.summary()
        return model


class RewardPredictor(object):
    '''
    '''
    def __init__(self, pref_q, weight_q, mgr_conn, buffer_len):
        self._pref_q = pref_q
        self._weight_q = weight_q
        self._mgr_conn = mgr_conn
        self._q = deque(maxlen = buffer_len)
        self.model = RewardPredictorModel()
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
        # Outputs the current model weights to the weight connection
        try:
            self._weight_q.put_nowait(self.model.get_weights())
        except Full:
            # Process 1 hasn't gotten the weight yet, remove it and add updated weight
            try:
                self._weight_q.get_nowait()
            except Empty: # rare case if p1 gets the old weight before we have a chance
                pass
            self._output_model_weights()

    def _learn(self):
        # Learns from the buffer, iterating over the whole queue
        for triple in self._q:
            self.model.fit(triple)

    def _check_msgs(self):
        # Read manager signals
        msgs = []
        while self._mgr_conn.poll():
            msgs.append(self._mgr_conn.recv())

        for msg in msgs:
            if msg.title == "stop":
                self._stop()

    def _stop(self):
        # Stop process execution
        self._stop_sig = True
        self._weight_q.close()                
        print(f"Quitting {current_process().name} process")
        os._exit(0)

    def _run(self):
        # Main process loop
        while True:
            self._check_msgs()
            self._get_comparisons()
            self._learn()
            self._output_model_weights()

def run_reward_predictor(pref_q, weight_q, mgr_conn):
    reward_predictor = RewardPredictor(pref_q=pref_q, weight_q=weight_q, mgr_conn=mgr_conn, buffer_len=BUFFER_LEN)
    reward_predictor._run()

if __name__ == "__main__":
    model = RewardPredictorModel()