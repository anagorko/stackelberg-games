"""
This module implements a game generator from

Paruchuri, Praveen, et al.
  "Playing games for security: An efficient exact algorithm for solving Bayesian Stackelberg games."
Proceedings of the 7th international joint conference on Autonomous agents and multiagent systems-Volume 2. 2008.

ARMOR generator - k patrols defending n terminals on an airport.
"""

from fractions import Fraction
from itertools import combinations
import typing

import numpy as np

from ..games.bayesian import BayesianGame
from .generator import GameGenerator, GameGeneratorMetadata


Numeric = typing.Union[int, float, Fraction]
NormalizedType = typing.TypeVar("NormalizedType", dict, tuple, Numeric)


def normalize(matrix: NormalizedType) -> NormalizedType:
    """Normalize values with linear map into [-1, 1] interval."""

    def r_max(x):
        if isinstance(x, dict):
            return max(r_max(v) for v in x.values())
        elif isinstance(x, typing.Iterable):
            return max(r_max(v) for v in x)
        else:
            return x

    def r_min(x):
        if isinstance(x, dict):
            return min(r_min(v) for v in x.values())
        elif isinstance(x, typing.Iterable):
            return min(r_min(v) for v in x)
        else:
            return x

    def r_map(x: NormalizedType, f) -> NormalizedType:
        if isinstance(x, dict):
            return {k: r_map(v, f) for k, v in x.items()}
        if isinstance(x, tuple):
            return tuple(r_map(v, f) for v in x)

        return f(x)

    a = r_min(matrix)
    b = r_max(matrix)

    if a < b:
        return r_map(matrix, lambda x: 2 * (x - a) / (b - a) - 1)
    else:
        return r_map(matrix, lambda _: 0)


class ARMORGenerator(GameGenerator):
    """ARMOR-canine problem instance generator."""

    def __init__(
        self,
        terminals: int,
        patrols: int,
        random_range: float = 0.1,
        norm=False,
        scale="C",
        randomize_attacker_payoff=False,
        shuffle_columns=True,
    ):
        self.terminals = terminals
        self.patrols = patrols
        self.random_range = random_range
        self.norm = norm
        self.scale = scale
        self.randomize_attacker_payoff = randomize_attacker_payoff
        self.shuffle_columns = shuffle_columns

    def get_metadata(self) -> GameGeneratorMetadata:
        """Serializes the Generator instance to a pydantic.BaseModel."""
        return GameGeneratorMetadata(
            cls=self.__class__,
            args=(),
            kwargs={
                "terminals": self.terminals,
                "patrols": self.patrols,
                "random_range": self.random_range,
                "norm": self.norm,
                "scale": self.scale,
                "randomize_attacker_payoff": self.randomize_attacker_payoff,
                "shuffle_columns": self.shuffle_columns,
            },
        )

    def _repr_html_(self) -> str:
        return (
            f"ARMOR-canine Bayesian Stackelberg game generator with {self.terminals} terminals "
            f"and {self.patrols} patrols."
        )

    def get_instance(self, random_seed: int) -> BayesianGame:
        """"""
        rng = np.random.default_rng(random_seed)

        def move_name(combination) -> str:
            return "".join(f"T_{t+1}" for t in combination)

        X = set(move_name(m) for m in combinations(range(self.terminals), self.patrols))
        T = {"L", "H"}
        p = {"L": 0.8, "H": 0.2}

        J = {
            t: set(f"T_{t+1}" for t in range(self.terminals)) | {r"\emptyset"}
            for t in sorted(T)
        }

        # terminal value
        if self.scale == "C":
            # constant
            V = {f"T_{t + 1}": 10 for t in range(self.terminals)}
        elif self.scale == "E":
            # exponential
            V = {f"T_{t + 1}": 5 * 2 ** (t + 1) for t in range(self.terminals)}
        elif self.scale == "L":
            # linear
            V = {f"T_{t + 1}": 10 * (t + 1) for t in range(self.terminals)}
        else:
            raise NotImplementedError("Choose C, E or L")

        # scale
        S = {"L": 1, "H": 5}

        R: dict[str, dict[str, dict[str, float]]] = {t: {} for t in sorted(T)}
        C: dict[str, dict[str, dict[str, float]]] = {t: {} for t in sorted(T)}

        for t in sorted(T):
            if self.shuffle_columns:
                permutation = rng.permutation(list(range(self.terminals)))
                V = {
                    f"T_{t + 1}": V[f"T_{permutation[t] + 1}"]
                    for t in range(self.terminals)
                }

            for i in sorted(X):
                R[t][i] = {}
                C[t][i] = {}

                for j in sorted(J[t], key=str):
                    if j == r"\emptyset":
                        R[t][i][j] = 0
                        C[t][i][j] = 0
                    else:
                        if j in i:
                            # terminal is defended
                            R[t][i][j] = S[t] * V[j] + rng.uniform(
                                -self.random_range * S[t] * V[j],
                                self.random_range * S[t] * V[j],
                            )
                            if self.randomize_attacker_payoff:
                                C[t][i][j] = -(
                                    S[t] * V[j]
                                    + rng.uniform(
                                        -self.random_range * S[t] * V[j],
                                        self.random_range * S[t] * V[j],
                                    )
                                )
                            else:
                                C[t][i][j] = -S[t] * V[j]
                        else:
                            R[t][i][j] = -S[t] * V[j] - rng.uniform(
                                -self.random_range * S[t] * V[j],
                                self.random_range * S[t] * V[j],
                            )
                            if self.randomize_attacker_payoff:
                                C[t][i][j] = S[t] * V[j] + rng.uniform(
                                    -self.random_range * S[t] * V[j],
                                    self.random_range * S[t] * V[j],
                                )
                            else:
                                C[t][i][j] = S[t] * V[j]

        if self.norm:
            R = normalize(R)
            for t in T:
                C[t] = normalize(C[t])

        return BayesianGame.from_dicts(
            typing.cast(set[str | int], X),
            typing.cast(set[str | int], T),
            typing.cast(dict[str | int, set[str | int]], J),
            typing.cast(dict[str | int, dict[str | int, dict[str | int, float]]], R),
            typing.cast(dict[str | int, dict[str | int, dict[str | int, float]]], C),
            typing.cast(dict[str | int, float], p),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ARMORGenerator):
            return False

        return (
            self.terminals == other.terminals
            and self.patrols == other.patrols
            and self.random_range == other.random_range
            and self.norm == other.norm
            and self.scale == other.scale
            and self.randomize_attacker_payoff == other.randomize_attacker_payoff
            and self.shuffle_columns == other.shuffle_columns
        )
