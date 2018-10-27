"""
A pure implementation of the Monte Carlo Tree Search

@author: Junxiao Song
modified by: Blaise Wang
"""

import copy
from operator import itemgetter

import numpy as np

from game import Board
from mcts_treenode import TreeNode


def roll_out_policy_func(board: 'Board'):
    """roll_out_policy_func -- a coarse, fast version of policy_fn used in the roll out phase."""
    # roll out randomly
    return zip(board.get_available_moves(), np.random.rand(len(board.get_available_moves())))


def policy_value_func(board: 'Board'):
    """a function that takes in a state and outputs a list of (action, probabilities)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    return zip(board.get_available_moves(),
               np.ones(len(board.get_available_moves())) / len(board.get_available_moves())), 0


class MCTS:
    """A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_function, c_puct=5, n_play_out=10000):
        """Arguments:
        policy_value_func -- a function that takes in a board state and outputs a list of (action, probabilities)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from 
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_function
        self._c_puct = c_puct
        self._n_play_out = n_play_out

    def _play_out(self, state: 'Board'):
        """Run a single play out from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self._root
        while 1:
            if node.is_leaf():
                break
                # Greedily select next move.
            action, node = node.select(self._c_puct)
            x, y = state.move_to_location(action)
            state.add_move(x, y)

        action_probabilities, _ = self._policy(state)
        # Check for end of game
        if state.has_winner() != -1:
            node.expand(action_probabilities)
        # Evaluate the leaf node by random roll out
        leaf_value = self._evaluate_roll_out(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    @staticmethod
    def _evaluate_roll_out(state: 'Board', limit=1000):
        """Use the roll out policy to play until the end of the game, returning +1 if the current
        player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        winner = -1
        player = state.get_current_player()
        for i in range(limit):
            winner = state.has_winner()
            if winner == -1:
                break
            max_action = max(roll_out_policy_func(state), key=itemgetter(1))[0]
            x, y = state.move_to_location(max_action)
            state.add_move(x, y)
        if winner == 0:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state: 'Board'):
        """Runs all play outs sequentially and returns the most visited action.
        Arguments:
        state -- the current state, including both game state and the current player.
        Returns:
        the selected action
        """
        [self._play_out(copy.deepcopy(state)) for _ in range(self._n_play_out)]
        return max(self._root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]

    def update_with_move(self, last_move: int):
        """Step forward in the tree, keeping everything we already know about the subtree.
        """
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSPlayer:
    """AI player based on MCTS"""

    def __init__(self, c_puct=5, n_play_out=2000):
        self.mcts = MCTS(policy_value_func, c_puct, n_play_out)

    def get_action(self, board: 'Board'):
        if board.get_move_number() < board.n * board.n:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
