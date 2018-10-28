"""
An implementation of the policyValueNet with TensorFlow

@author: Sheng Liang
modified by: Blaise Wang
"""

import tensorflow as tf
import numpy as np

from game import Board


class PolicyValueNet:
    """policy-value network """

    def __init__(self, n: int, model_file=None):
        self.n = n
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.l2_const = 1e-4  # coefficient of l2 penalty

        """create the policy value network """
        self.input_state = tf.placeholder(tf.float32, shape=[None, 4, self.n, self.n], name='state')
        self.input_states = tf.transpose(self.input_state, [0, 2, 3, 1])
        self.winner = tf.placeholder(tf.float32, shape=[None, 1], name='winner')
        self.mcts_probabilities = tf.placeholder(tf.float32, shape=[None, self.n * self.n], name='mcts_probabilities')

        """
        convolutional layers
        """
        self.network = tf.layers.conv2d(inputs=self.input_states, filters=32, kernel_size=[3, 3], padding="same",
                                        data_format="channels_last", activation=tf.nn.relu)
        self.network = tf.layers.conv2d(inputs=self.network, filters=64, kernel_size=[3, 3], padding="same",
                                        data_format="channels_last", activation=tf.nn.relu)
        self.network = tf.layers.conv2d(inputs=self.network, filters=128, kernel_size=[3, 3], padding="same",
                                        data_format="channels_last", activation=tf.nn.relu)

        """
        action policy layers
        """
        self.policy_net = tf.layers.conv2d(inputs=self.network, filters=4, kernel_size=[1, 1], padding="same",
                                           data_format="channels_last", activation=tf.nn.relu)
        self.policy_net = tf.reshape(self.policy_net, [-1, 4 * self.n * self.n])
        self.policy_net = tf.layers.dense(inputs=self.policy_net, units=self.n * self.n, activation=tf.nn.log_softmax)

        """
        state value layers
        """
        self.value_net = tf.layers.conv2d(inputs=self.network, filters=2, kernel_size=[1, 1], padding="same",
                                          data_format="channels_last", activation=tf.nn.relu)

        self.value_net = tf.reshape(self.value_net, [-1, 2 * self.n * self.n])
        self.value_net = tf.layers.dense(inputs=self.value_net, units=64, activation=tf.nn.relu)
        self.value_net = tf.layers.dense(inputs=self.value_net, units=1, activation=tf.nn.tanh)

        """
        Loss function
        Three loss terms：
        loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        """
        # 1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.winner, self.value_net)
        # 2. Policy Loss function
        self.policy_loss = tf.negative(
            tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probabilities, self.policy_net), 1)))
        # 3. L2 penalty
        variables = tf.trainable_variables()
        l2_penalty = self.l2_const * tf.add_n([tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name.lower()])
        # 4 Add up
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.policy_net) * self.policy_net, 1)))
        # Make a session and initialize variables
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)

        # Saving and restoring
        self.saver = tf.train.Saver()
        if model_file:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probabilities, value = self.session.run([self.policy_net, self.value_net],
                                                        feed_dict={self.input_state: state_batch})
        act_probabilities = np.exp(log_act_probabilities)
        return act_probabilities, value

    def train_step(self, state_batch, mcts_probabilities, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
            [self.loss, self.entropy, self.optimizer],
            feed_dict={self.input_state: state_batch, self.mcts_probabilities: mcts_probabilities,
                       self.winner: winner_batch, self.learning_rate: lr})
        return loss, entropy

    def policy_value_func(self, board: 'Board'):
        """
        input: board
        output: a list of (action, probabilities) tuples for each available action and the score of the board state
        """
        legal_positions = board.get_available_moves(board.get_current_player())
        current_state = np.ascontiguousarray(board.get_current_state().reshape(-1, 4, self.n, self.n))
        act_probabilities, value = self.policy_value(current_state)
        act_probabilities = zip(legal_positions, act_probabilities.flatten()[legal_positions])
        return act_probabilities, value

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
