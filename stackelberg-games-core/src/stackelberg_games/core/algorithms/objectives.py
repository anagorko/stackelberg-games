from ..games.bayesian import BayesianGame
import numpy as np
from abc import ABC, abstractmethod


def x_to_dict(x, objective):
    return {objective.X_sorted[i]: x[i] for i in range(0, len(x))}


class StackelbergObjective(ABC):
    def __init__(self, game: BayesianGame.VerboseModel, dims, turn, ub):
        self.game = game
        self.X_sorted = list(self.game.X)
        self.counter = 0
        self.dims = dims
        self.lb = np.zeros(dims)
        self.ub = np.full(dims, ub)
        self.turn = turn

    @abstractmethod
    def projection(self, p):
        pass

    def __call__(self, x):
        self.counter += 1
        x_dict = x_to_dict(self.projection(x), self)
        return self.game.expected_reward(x_dict)


class StackelbergBasicObjective(StackelbergObjective):
    def __init__(self, game: BayesianGame.VerboseModel, dims=1, turn=0.1):
        super().__init__(game, dims, turn, ub=1)

    def projection(self, p):
        return p


class StackelbergRadialProjectionObjective(StackelbergObjective):
    def __init__(self, game: BayesianGame.VerboseModel, dims=1, turn=0.1, ub=50):
        super().__init__(game, dims, turn, ub)

    def projection(self, p):
        x = np.zeros(self.dims)
        if all(v == 0 for v in p):
            points = np.zeros(len(p))
        else:
            points = (np.array(p) + x) / (sum(p) + sum(x))
        return points


class StackelbergSoftmaxObjective(StackelbergObjective):
    def __init__(self, game: BayesianGame.VerboseModel, dims=1, turn=0.1, ub=30):
        super().__init__(game, dims, turn, ub)

    def projection(self, p):
        beta = 0.1
        points = np.exp(beta * np.array(p)) / np.sum(np.exp(beta * np.array(p)))
        return points
