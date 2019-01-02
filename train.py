"""
An implementation of the training pipeline of AlphaZero

@author: Junxiao Song
modified by: Blaise Wang
"""

from collections import defaultdict, deque
import os
import multiprocessing
import random
import sys
import time

import numpy as np

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from mcts import MCTSPlayer as MCTS_Pure
from policy_value_net import PolicyValueNet


def print_log(string: str):
    with open("log", 'a') as file:
        file.write(string + "\n")
    file.close()


def data_log(string: str):
    with open("data.log", 'a') as file:
        file.write(string + "\n")
    file.close()


class TrainPipeline:
    def __init__(self, init_model=None):
        # params of the board and the game
        self.n = 8
        self.board = Board(self.n)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 5e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_play_out = 400  # number of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5  # number of train_steps for each update
        self.kl_target = 0.025
        self.check_freq = 50
        self.game_batch_number = 10000
        self.best_win_ratio = 0.0
        self.episode_length = 0
        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        # number of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        self.last_batch_number = 0
        self.pure_mcts_play_out_number = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.n, model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.n)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_func, c_puct=self.c_puct,
                                      n_play_out=self.n_play_out, is_self_play=1)

    def get_equivalent_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]"""
        extend_data = []
        for state, mcts_probabilities, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equivalent_state = np.array([np.rot90(s, i) for s in state])
                equivalent_mcts_prob = np.rot90(np.flipud(mcts_probabilities.reshape(self.n, self.n)), i)
                extend_data.append((equivalent_state, np.flipud(equivalent_mcts_prob).flatten(), winner))
                # flip horizontally
                equivalent_state = np.array([np.fliplr(s) for s in equivalent_state])
                equivalent_mcts_prob = np.fliplr(equivalent_mcts_prob)
                extend_data.append((equivalent_state, np.flipud(equivalent_mcts_prob).flatten(), winner))
        return extend_data

    def collect_self_play_data(self):
        """collect self-play data for training"""
        play_data = list(self.game.start_self_play(self.mcts_player, temp=self.temp))
        self.episode_length = len(play_data)
        play_data = self.get_equivalent_data(play_data)
        self.data_buffer.extend(play_data)

    def collect_play_data(self, data):
        """collect self-play data for training"""
        play_data = list(data)
        self.episode_length = len(play_data)
        play_data = self.get_equivalent_data(play_data)
        self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        kl = 0
        new_v = 0
        loss = 0
        entropy = 0

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probabilities_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probabilities, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probabilities_batch, winner_batch,
                                                             self.learn_rate * self.lr_multiplier)
            new_probabilities, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(old_probabilities * (np.log(old_probabilities + 1e-10) - np.log(new_probabilities + 1e-10)),
                       axis=1))
            if kl > self.kl_target * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_target * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_target / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        print_log("kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".
                  format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing games against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_func, c_puct=self.c_puct,
                                         n_play_out=self.n_play_out)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_play_out=self.pure_mcts_play_out_number)
        win_cnt = defaultdict(int)
        results = self.pool.map(self.game.start_play,
                                [(current_mcts_player, pure_mcts_player, i) for i in range(n_games)])
        for winner in results:
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print_log("number_play_outs:{}, win: {}, lose: {}, tie:{}".
                  format(self.pure_mcts_play_out_number, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self, data=None):
        """run the training pipeline"""
        try:
            if data:
                for each_data in data:
                    self.collect_play_data(each_data)
                if len(self.data_buffer) > self.batch_size:
                    _, _ = self.policy_update()
                    self.policy_value_net.save_model('./new_policy.model')
                for i in range(self.game_batch_number):
                    if os.path.exists("done"):
                        break
                    start_time = time.time()
                    self.collect_self_play_data()
                    print_log(
                        "batch i:{}, episode_len:{}, in:{}".format(i + 1 + self.last_batch_number, self.episode_length,
                                                                   time.time() - start_time))
                    if len(self.data_buffer) > self.batch_size:
                        loss, entropy = self.policy_update()
                        data_log(str((i + 1 + self.last_batch_number, loss, entropy)))
                    if (i + 1) % self.check_freq == 0:
                        print_log("current self-play batch: {}".format(i + 1 + self.last_batch_number))
                        self.policy_value_net.save_model('./current_policy.model')

        except KeyboardInterrupt:
            pass


sys.setrecursionlimit(256 * 256)
training_pipeline = TrainPipeline()
training_pipeline.run()
