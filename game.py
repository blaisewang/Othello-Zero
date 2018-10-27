"""
An implementation of the game and the game board

@author: Blaise Wang
"""

import numpy as np

from mcts_alphaZero import MCTSPlayer


def to_list(p1: int, p2: int, n: int) -> []:
    c = [((p1 ^ (p1 >> 1)) >> k) & 1 for k in range(0, n >> 1)][::-1]
    return c + [((p2 ^ (p2 >> 1)) >> k) & 1 for k in range(0, n - (n >> 1))][::-1]


class Board:
    def __init__(self, n: int):
        self.n = n
        self.winner = -1
        self.move_list = []
        self.chess = np.repeat(0, self.n * self.n).reshape(self.n, self.n)

    def initialize(self):
        self.winner = -1
        self.move_list = []
        self.chess[0:self.n, 0:self.n] = 0

    def add_move(self, x: int, y: int):
        self.move_list.append((x, y))
        self.chess[x, y] = 2 if self.get_move_number() % 2 == 0 else 1

    def remove_move(self):
        x, y = self.move_list.pop()
        self.chess[x, y] = 0

    def move_to_location(self, move: int) -> (int, int):
        x = self.n - move // self.n - 1
        y = move % self.n
        return x, y

    def location_to_move(self, x: int, y: int) -> int:
        return (self.n - x - 1) * self.n + y

    def get_available_moves(self) -> []:
        potential_move_list = []
        for (x, y), value in np.ndenumerate(self.chess):
            if not value:
                potential_move_list.append(self.location_to_move(x, y))
        return sorted(potential_move_list)

    def get_current_state(self):
        player = self.get_current_player()
        opponent = 2 if player == 1 else 1
        square_state = np.zeros((4, self.n, self.n))
        for (x, y), value in np.ndenumerate(self.chess):
            if value == player:
                square_state[0][self.n - x - 1][y] = 1.0
            elif value == opponent:
                square_state[1][self.n - x - 1][y] = 1.0
        if self.get_move_number() > 0:
            x, y = self.move_list[self.get_move_number() - 1]
            square_state[2][self.n - x - 1][y] = 1.0
        if self.get_current_player() == 1:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]

    def get_move_number(self) -> int:
        return len(self.move_list)

    def get_current_player(self) -> int:
        return 1 if self.get_move_number() % 2 == 0 else 2

    def has_winner(self):

        """
        黑胜返回1
        白胜返回2
        平局返回0
        未结束返回-1
        """

        return 0


class Game:
    def __init__(self, board: 'Board'):
        self.board = board

    def start_play(self, args) -> int:
        player1, player2, index = args
        if index % 2:
            player1, player2 = player2, player1
        self.board.initialize()
        while self.board.get_move_number() < self.board.n * self.board.n:
            player_in_turn = player1 if self.board.get_current_player() == 1 else player2
            move = player_in_turn.get_action(self.board)
            x, y = self.board.move_to_location(move)
            self.board.add_move(x, y)
            winner = self.board.has_winner()
            if winner != -1:
                if not winner:
                    return winner
                if index % 2:
                    return 1 if winner == 2 else 2

    def start_self_play(self, player: 'MCTSPlayer', temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree
        store the self-play data: (state, mcts_probabilities, z)
        """
        self.board.initialize()
        states, mcts_probabilities, current_players = [], [], []
        while self.board.get_move_number() < self.board.n * self.board.n:
            move, move_probabilities = player.get_action(self.board, temp=temp, return_probability=1)
            # store the data
            states.append(self.board.get_current_state())
            mcts_probabilities.append(move_probabilities)
            current_players.append(self.board.get_current_player())
            # perform a move
            x, y = self.board.move_to_location(move)
            self.board.add_move(x, y)
            winner = self.board.has_winner()
            if winner != -1:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                return zip(states, mcts_probabilities, winners_z)
