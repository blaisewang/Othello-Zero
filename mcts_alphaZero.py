"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value network
to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
modified by: Blaise Wang
"""

import copy

import numpy as np

from mcts_treenode import TreeNode


def soft_max(x):
    probabilities = np.exp(x - np.max(x))
    probabilities /= np.sum(probabilities)
    return probabilities


class MCTS:
    """A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_func, c_puct=5, n_play_out=10000):
        """Arguments:
        policy_value_func -- a function that takes in a board state and outputs a list of (action, probabilities)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from 
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_func
        self._c_puct = c_puct
        self._n_play_out = n_play_out

    def _play_out(self, state):
        """Run a single play_out from the root to the leaf, getting a value at the leaf and
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

        # Evaluate the leaf using a network which outputs a list of (action, probabilities)
        # tuples p and also a score v in [-1, 1] for the current player.
        action_probabilities, leaf_value = self._policy(state)
        # Check for end of game.
        winner = state.has_winner()
        if not winner != -1:
            node.expand(action_probabilities)
        else:
            # for end state，return the "true" leaf_value
            if winner == 0:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probabilities(self, state, temp=1e-3):
        """Runs all play_outs sequentially and returns the available actions and their corresponding probabilities
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities 
        """
        [self._play_out(copy.deepcopy(state)) for _ in range(self._n_play_out)]
        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node.n_visits) for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probabilities = soft_max(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probabilities

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

    def __init__(self, policy_value_function, c_puct=5, n_play_out=2000, is_self_play=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_play_out)
        self._is_self_play = is_self_play

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_probability=0):
        move_probabilities = np.zeros(board.n * board.n)
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        if board.get_move_number() < board.n * board.n:
            acts, probabilities = self.mcts.get_move_probabilities(board, temp)
            if return_probability == 2:
                return acts, probabilities
            move_probabilities[list(acts)] = probabilities
            if self._is_self_play:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(acts, p=0.70 * probabilities + 0.30 * np.random.dirichlet(
                    0.3 * np.ones(len(probabilities))))
                self.mcts.update_with_move(move)  # update the root node and reuse the search tree
            else:
                # with the default temp, this is almost equivalent to choosing the move with the highest probability
                move = np.random.choice(acts, p=probabilities)
                # reset the root node
                self.mcts.update_with_move(-1)
            if return_probability == 1:
                return move, move_probabilities
            else:
                return move
