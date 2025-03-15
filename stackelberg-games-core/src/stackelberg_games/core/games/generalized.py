"""
This module implements a data class for generalized Bayesian games.
"""

from __future__ import annotations
from typing import Sequence, cast

# noinspection PyPackageRequirements
import cdd  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt

from .bayesian import BayesianGame
from .game import Game
from .normal_form import NormalFormGame
from ..linalg import affine_transformation_matrix, to_matrix
from ..polyhedra import m_simplex, preimage


class GeneralizedNormalFormGame(Game):
    """
    A generalized normal form game.
    """

    def __init__(
        self,
        defender_payoff: npt.ArrayLike | npt.NDArray[np.float64],
        attacker_payoff: npt.ArrayLike | npt.NDArray[np.float64],
        defender_mixed_strategies: cdd.Polyhedron,
        attacker_moves: Sequence[str | int] | None = None,
        defender_name: str | int | None = None,
        attacker_name: str | int | None = None,
    ) -> None:
        self.defender_payoff = to_matrix(defender_payoff)
        self.attacker_payoff = to_matrix(attacker_payoff)

        assert self.defender_payoff.ndim == 2
        assert self.defender_payoff.shape == self.attacker_payoff.shape

        self.defender_mixed_strategies = defender_mixed_strategies

        if attacker_moves is None:
            self.attacker_moves: tuple[str | int, ...] = tuple(
                range(self.attacker_payoff.shape[1])
            )
        else:
            self.attacker_moves = tuple(attacker_moves)

        if defender_name is None:
            self.defender_name: str | int = "Alice"
        else:
            self.defender_name = defender_name

        if attacker_name is None:
            self.attacker_name: str | int = "Bob"
        else:
            self.attacker_name = attacker_name

    @classmethod
    def from_standard_game(cls, other: NormalFormGame) -> GeneralizedNormalFormGame:
        """
        Returns GeneralizedNormalFormGame instance created from a NormalFormGame instance.
        """
        defender_mixed_strategies = m_simplex(len(other.defender_moves))
        return cls(
            np.vstack(
                (
                    other.defender_payoff,
                    np.zeros((1, other.defender_payoff.shape[1]), dtype=np.float64),
                )
            ),
            np.vstack(
                (
                    other.attacker_payoff,
                    np.zeros((1, other.attacker_payoff.shape[1]), dtype=np.float64),
                )
            ),
            defender_mixed_strategies,
            other.attacker_moves,
            other.defender_name,
            other.attacker_name,
        )

    def preimage(
        self, augmented_matrix: npt.NDArray[np.float64]
    ) -> GeneralizedNormalFormGame:
        """
        Returns a restricted game.
        """

        mixed_strategies_preimage = preimage(
            augmented_matrix, self.defender_mixed_strategies
        )
        affine_matrix = affine_transformation_matrix(
            augmented_matrix[:, :-1], augmented_matrix[:, -1]
        )

        return self.__class__(
            affine_matrix.T @ self.defender_payoff,
            affine_matrix.T @ self.attacker_payoff,
            mixed_strategies_preimage,
            self.attacker_moves,
            self.defender_name,
            self.attacker_name,
        )

    def response_inducing_region(self, response: int) -> cdd.Polyhedron:
        region = self.defender_mixed_strategies.get_inequalities()

        for j in range(self.attacker_payoff.shape[1]):
            if j == response:
                continue
            # [x 1] C^t e_{response_t} >= [x 1] C^t e_j
            # [x 1] (C^t e_{response_t} - C^t e_j) >= 0

            row = self.attacker_payoff[:, response] - self.attacker_payoff[:, j]
            region.extend([np.hstack(([row[-1]], row[:-1]))], linear=False)

        return cdd.Polyhedron(region)


class GeneralizedBayesianGame(Game):
    """
    A generalized Bayesian game.
    """

    def __init__(
        self,
        defender_payoffs: Sequence[npt.ArrayLike | npt.NDArray[np.float64]],
        attacker_payoffs: Sequence[npt.ArrayLike] | npt.NDArray[np.float64],
        attacker_distribution: Sequence[float | np.float64] | npt.NDArray[np.float64],
        defender_mixed_strategies: cdd.Polyhedron,
        attackers_moves: Sequence[Sequence[str | int]],
        defender_name: str | int | None = None,
        attacker_names: Sequence[str | int] | None = None,
    ) -> None:
        self.defender_payoffs = tuple(
            to_matrix(defender_payoff) for defender_payoff in defender_payoffs
        )
        self.attacker_payoffs = tuple(
            to_matrix(attacker_payoff) for attacker_payoff in attacker_payoffs
        )

        assert all(
            defender_payoff.ndim == 2 for defender_payoff in self.defender_payoffs
        )
        assert all(
            defender_payoff.shape == attacker_payoff.shape
            for defender_payoff, attacker_payoff in zip(
                self.defender_payoffs, self.attacker_payoffs
            )
        )

        self.attacker_distribution: npt.NDArray[np.float64] = np.array(
            attacker_distribution, dtype=np.float64
        )
        self.defender_mixed_strategies = defender_mixed_strategies

        self.attackers_moves = tuple(
            tuple(attacker_moves) for attacker_moves in attackers_moves
        )

        if defender_name is None:
            self.defender_name: str | int = "Alice"
        else:
            self.defender_name = defender_name

        if attacker_names is None:
            self.attacker_names: tuple[str | int, ...] = tuple(
                f"Bob{i}" for i in range(len(attacker_distribution))
            )
        else:
            self.attacker_names = tuple(attacker_names)

    @classmethod
    def from_standard_game(cls, other: BayesianGame) -> GeneralizedBayesianGame:
        """
        Returns GeneralizedBayesianGame instance created from a BayesianGame instance.
        """
        defender_mixed_strategies = m_simplex(len(other.defender_moves))
        return cls(
            tuple(
                np.vstack(
                    (
                        defender_payoff,
                        np.zeros((1, defender_payoff.shape[1]), dtype=np.float64),
                    )
                )
                for defender_payoff in other.defender_payoffs
            ),
            tuple(
                np.vstack(
                    (
                        attacker_payoff,
                        np.zeros((1, attacker_payoff.shape[1]), dtype=np.float64),
                    )
                )
                for attacker_payoff in other.attacker_payoffs
            ),
            other.attacker_distribution,
            defender_mixed_strategies,
            other.attackers_moves,
            other.defender_name,
            other.attacker_names,
        )

    def preimage(
        self, augmented_matrix: npt.NDArray[np.float64]
    ) -> GeneralizedBayesianGame:
        """
        Returns a restricted game. Embedding Ax+b is represented by an augmented matrix [A | b].
        """

        mixed_strategies_preimage = preimage(
            augmented_matrix, self.defender_mixed_strategies
        )
        affine_matrix = affine_transformation_matrix(
            augmented_matrix[:, :-1], augmented_matrix[:, -1]
        )

        return self.__class__(
            tuple(
                affine_matrix.T @ defender_payoff
                for defender_payoff in self.defender_payoffs
            ),
            tuple(
                affine_matrix.T @ attacker_payoff
                for attacker_payoff in self.attacker_payoffs
            ),
            self.attacker_distribution,
            mixed_strategies_preimage,
            self.attackers_moves,
            self.defender_name,
            self.attacker_names,
        )

    def response_inducing_region(self, response: tuple[int, ...]) -> cdd.Polyhedron:
        region = self.defender_mixed_strategies.get_inequalities()

        for t in range(self.attacker_distribution.shape[0]):
            for j in range(self.attacker_payoffs[t].shape[1]):
                if j == response[t]:
                    continue
                # [x 1] C^t e_{response_t} >= [x 1] C^t e_j
                # [x 1] (C^t e_{response_t} - C^t e_j) >= 0

                row = (
                    self.attacker_payoffs[t][:, response[t]]
                    - self.attacker_payoffs[t][:, j]
                )
                region.extend([np.hstack(([row[-1]], row[:-1]))], linear=False)

        return cdd.Polyhedron(region)

    @property
    def m(self) -> int:
        return self.defender_payoffs[0].shape[0]

    @property
    def T(self) -> int:
        return len(self.attacker_distribution)

    @property
    def n(self) -> tuple[int, ...]:
        return tuple(self.defender_payoffs[i].shape[1] for i in range(self.T))

    def correlation_coefficient(self) -> np.float64:
        """Computes a correlation coefficient between defender and attacker payoff matrices."""

        return cast(
            np.float64,
            sum(
                self.attacker_distribution[t]
                * (
                    np.dot(
                        self.defender_payoffs[t].flatten(),
                        self.attacker_payoffs[t].flatten(),
                    )
                )
                / np.linalg.norm(self.defender_payoffs[t].flatten())
                / np.linalg.norm(self.attacker_payoffs[t].flatten())
                for t in range(self.T)
            ),
        )
