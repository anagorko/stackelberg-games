# MIT License
#
# Copyright (c) 2024 Łukasz Gołuchowski
# Copyright (c) 2024 Bop2000
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from typing import Union
from fractions import Fraction


Numeric = Union[int, float, Fraction]


class BayesianStackelbergSolution:
    """"""

    def __init__(self, x: dict, q: dict, xi: Numeric):
        """"""
        self.x = x
        self.q = q
        self.xi = xi


################################# DOTS algorithm ################################


class DOTS:
    def __init__(
        self,
        exploration_weight=None,
        f=None,
        name=None,
        sum_to_one=True,
        task_klass=None,
        random_nodes_factor=2,
        most_visited_nodes_n=1,
        top_nodes_n=5,
        random_nodes_n=1,
        seed=None,
    ):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.f = f
        self.name = name
        self.sum_to_one = sum_to_one
        self.task_klass = task_klass
        self.random_nodes_factor = random_nodes_factor
        self.most_visited_nodes_n = most_visited_nodes_n
        self.top_nodes_n = top_nodes_n
        self.random_nodes_n = random_nodes_n
        self.rng = np.random.default_rng(seed)

    def _uct_node(self, node):
        """Upper confidence bound for trees"""

        def uct(n):
            log_N_vertex = math.log(self.N[node])

            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n] + 1)
            )
            return uct_value

        return uct

    def _generate_actions(self, node):
        if self.sum_to_one:
            action = [
                (p1, p2)
                for p1 in range(0, len(node.tup))
                for p2 in range(p1 + 1, len(node.tup))
            ]
        else:
            action = [p for p in range(0, len(node.tup))]
        return action

    def choose(self, node):
        """Choose the best successor of node."""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        action = self._generate_actions(node)

        self.children[node] = node.find_children(action, self.f, self.rng)

        uct = self._uct_node(node)

        media_node = max(self.children[node], key=uct)
        node_rand = []
        # for i in range(len(list(self.children[node]))):
        ind = self.rng.integers(
            0, len(list(self.children[node])), self.random_nodes_factor
        )  # #for computer memory consideration, choose only 2 random
        # nodes in one rollout
        for i in ind:
            node_rand.append(list(self.children[node])[i].tup)

        if uct(media_node) > uct(node):
            return media_node, node_rand
        return node, node_rand

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def data_process(self, X, boards):
        new_x = []
        boards = np.array(boards)
        boards = np.unique(boards, axis=0)
        for i in boards:
            temp_x = np.array(i)
            same = np.all(temp_x == X, axis=1)
            has_true = any(same)
            if not has_true:
                new_x.append(temp_x)
        new_x = np.array(new_x)
        return new_x

    def most_visit_node(self, X, top_n):
        N_visit = self.N
        children = [i for i in self.children]
        children_N = []
        X_top = []
        for child in children:
            child_tup = np.array(child.tup)
            same = np.all(child_tup == X, axis=1)
            has_true = any(same)
            if not has_true:
                children_N.append(N_visit[child])
                X_top.append(child_tup)
        children_N = np.array(children_N)
        X_top = np.array(X_top)
        ind = np.argpartition(children_N, -top_n)[-top_n:]
        X_topN = X_top[ind]
        return X_topN

    def single_rollout(self, X, rollout_round, board_uct):
        boards = []
        boards_rand = []
        for i in range(0, rollout_round):
            self.do_rollout(board_uct)
            board_uct, board_rand = self.choose(board_uct)
            boards.append(list(board_uct.tup))
            boards_rand.append(list(board_rand))

        # visit nodes
        X_most_visit = self.most_visit_node(X, self.most_visited_nodes_n)

        # highest pred value nodes and random nodes
        new_x = self.data_process(X, boards)

        new_pred = [self.f(x) for x in new_x]
        new_pred = np.array(new_pred).reshape(len(new_x))

        boards_rand = np.vstack(boards_rand)
        new_rands = self.data_process(X, boards_rand)
        top_n = self.top_nodes_n
        if len(new_x) >= top_n:
            ind = np.argsort(new_pred)[-top_n:]
            top_X = new_x[ind]
            X_rand2 = [
                new_rands[self.rng.integers(0, len(new_rands) - 1, endpoint=True)]
                for _ in range(self.random_nodes_n)
            ]
        elif len(new_x) == 0 and len(new_rands) >= top_n:
            new_pred = np.array([self.f(x) for x in new_rands]).reshape(-1)
            ind = np.argsort(new_pred)[-top_n:]
            top_X = new_rands[ind]
            X_rand2 = [
                new_rands[self.rng.integers(0, len(new_rands) - 1, endpoint=True)]
                for _ in range(self.random_nodes_n)
            ]
        else:
            top_X = np.array(new_x)
            num_random = self.top_nodes_n + self.random_nodes_n - len(top_X)
            if len(new_rands) == 0:
                X_rand2 = []
            else:
                X_rand2 = [
                    new_rands[self.rng.integers(0, len(new_rands) - 1, endpoint=True)]
                    for _ in range(num_random)
                ]

        non_empty = list()
        if len(X_most_visit) > 0:
            non_empty.append(X_most_visit)
        if len(top_X) > 0:
            non_empty.append(top_X)
        if len(X_rand2) > 0:
            non_empty.append(X_rand2)

        top_X = np.concatenate(non_empty)

        return top_X

    def rollout(self, X, y, rollout_round, ratio, iteration):
        if iteration % 100 < 80:
            UCT_low = False
        else:
            UCT_low = True

        #### make sure unique initial points
        ind = np.argsort(y)
        x_current_top = X[ind[-3:]]
        indexes = np.unique(x_current_top, axis=0, return_index=True)[1]
        x_current_top = np.array(
            [x_current_top[index] for index in sorted(indexes, reverse=True)]
        )
        i = 0
        while len(x_current_top) < 3:
            x_current_top = np.concatenate(
                (
                    x_current_top.reshape(-1, self.f.dims),
                    X[ind[i - 4]].reshape(-1, self.f.dims),
                ),
                axis=0,
            )
            i -= 1
            x_current_top = np.unique(x_current_top, axis=0)

        ### starting rollout
        X_top = []
        for i in range(3):
            initial_X = x_current_top[i]
            values = max(y)
            exp_weight = ratio * abs(values)
            if UCT_low:
                values = self.f(initial_X)
                exp_weight = ratio * 0.5 * values
            self.exploration_weight = exp_weight
            board_uct = self.task_klass(
                tup=tuple(initial_X), value=values, terminal=False
            )
            top_X = self.single_rollout(X, rollout_round, board_uct)
            X_top.append(top_X)

        top_X = np.vstack(X_top)
        top_X = top_X[:20]
        return top_X

    def _select(self, node):
        """Find an unexplored descendent of `node`"""
        path = []
        count = 0
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()

            def evaluate(n):
                return n.value

            if count == 50:
                return path
            if unexplored:
                path.append(max(unexplored, key=evaluate))  #
                return path
            node = self._uct_select(node)  # descend a layer deeper
            count += 1

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        if node in self.children:
            return  # already expanded
        action = self._generate_actions(node)
        self.children[node] = node.find_children(action, self.f, self.rng)

    def _simulate(self, node):
        """Returns the reward for a random simulation (to completion) of `node`"""
        reward = node.reward(self.f)
        return reward

    def _backpropagate(self, path, reward):
        """Send the reward back up to the ancestors of the leaf"""
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        """Select a child of node, balancing exploration & exploitation"""
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        uct = self._uct_node(node)

        uct_node = max(self.children[node], key=uct)
        return uct_node


class Node(ABC):
    """
    A representation of a single board state.
    DOTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True


_OT = namedtuple("opt_task", "tup value terminal")  # type: ignore[name-match]


class opt_task(_OT, Node):
    def find_children(board, action, f, rng):
        if board.terminal:
            return set()
        turn = f.turn
        aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn).round(5)
        all_tup = []
        for index in action:
            tup = list(board.tup)
            flip = rng.integers(0, 5, endpoint=True)
            if flip == 0:
                tup[index] += turn
            elif flip == 1:
                tup[index] -= turn
            elif flip == 2:
                for i in range(int(f.dims / 5)):
                    index_2 = rng.integers(0, len(tup) - 1, endpoint=True)
                    tup[index_2] = rng.choice(aaa)
            elif flip == 3:
                for i in range(int(f.dims / 10)):
                    index_2 = rng.integers(0, len(tup) - 1, endpoint=True)
                    tup[index_2] = rng.choice(aaa)
            elif flip == 4:
                tup[index] = rng.choice(aaa)
            elif flip == 5:
                tup[index] = rng.choice(aaa)
            tup[index] = round(tup[index], 5)

            tup = np.array(tup)
            ind1 = np.where(tup > f.ub[0])[0]
            if len(ind1) > 0:
                tup[ind1] = f.ub[0]
            ind1 = np.where(tup < f.lb[0])[0]
            if len(ind1) > 0:
                tup[ind1] = f.lb[0]
            all_tup.append(tup)

        all_value = [f(tup) for tup in all_tup]
        is_terminal = False
        return {opt_task(tuple(t), v, is_terminal) for t, v in zip(all_tup, all_value)}

    def reward(board, f):
        return f(board.tup)

    def is_terminal(board):
        return board.terminal
