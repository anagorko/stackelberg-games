"""
This module implements generators for Bayesian games with random payoff matrices.
"""

import math

import numpy as np

from ..games.bayesian import BayesianGame
from .generator import GameGenerator, GameGeneratorMetadata


class RandomCorrelatedGameGenerator(GameGenerator):
    """ """

    tag: str = "random_correlated"

    def __init__(
        self,
        number_of_defender_moves: int,
        number_of_attacker_moves: int,
        number_of_attacker_types: int,
        correlation_coefficient: float = 0.0,
        uniform_attacker_distribution: bool = False,
    ) -> None:
        assert number_of_defender_moves * number_of_attacker_moves > 1

        self.number_of_defender_moves = number_of_defender_moves
        self.number_of_attacker_moves = number_of_attacker_moves
        self.number_of_attacker_types = number_of_attacker_types
        self.correlation_coefficient = correlation_coefficient
        self.uniform_attacker_distribution = uniform_attacker_distribution

    def get_metadata(self) -> GameGeneratorMetadata:
        """Serializes the Generator instance to a pydantic.BaseModel."""
        return GameGeneratorMetadata(
            cls=self.__class__,
            args=(),
            kwargs={
                "number_of_defender_moves": self.number_of_defender_moves,
                "number_of_attacker_moves": self.number_of_attacker_moves,
                "number_of_attacker_types": self.number_of_attacker_types,
                "correlation_coefficient": self.correlation_coefficient,
                "uniform_attacker_distribution": self.uniform_attacker_distribution,
            },
        )

    def get_instance(self, random_seed: int) -> BayesianGame:
        """"""
        rng = np.random.default_rng(random_seed)

        defender_payoffs = []
        attacker_payoffs = []

        for _ in range(self.number_of_attacker_types):
            dim = self.number_of_defender_moves * self.number_of_attacker_moves

            # Defender payoff matrix, sampled from the normal distribution
            df = rng.normal(size=dim)

            # A random matrix in the orthogonal complement of df sampled from the normal distribution

            # This is too slow:
            # embedding = np.linalg.qr(
            #     np.hstack((df.reshape((dim, 1)), np.identity(dim)))
            # ).Q[:, 1:]
            # att = embedding @ rng.normal(size=dim-1)

            tmp = rng.normal(size=dim)
            proj = (df @ tmp / np.inner(df, df)) * df
            att = tmp - proj

            defender_payoff = df.reshape(
                (self.number_of_defender_moves, self.number_of_attacker_moves)
            )
            orthogonal_payoff = att.reshape(
                (self.number_of_defender_moves, self.number_of_attacker_moves)
            )

            defender_payoff = defender_payoff
            orthogonal_payoff = (
                orthogonal_payoff
                / np.linalg.norm(orthogonal_payoff)
                * np.linalg.norm(defender_payoff)
            )

            defender_payoffs.append(defender_payoff)
            # attacker_payoffs.append(
            #    self.correlation_coefficient * defender_payoff
            #    + (1 - abs(self.correlation_coefficient)) * orthogonal_payoff
            # )
            attacker_payoffs.append(
                math.sqrt(1 - self.correlation_coefficient**2) * orthogonal_payoff
                + self.correlation_coefficient * defender_payoff
            )

        if self.uniform_attacker_distribution:
            attacker_distribution = [
                1 / self.number_of_attacker_types
            ] * self.number_of_attacker_types
        else:
            attacker_distribution = rng.dirichlet(
                (1,) * self.number_of_attacker_types
            ).tolist()

        return BayesianGame(defender_payoffs, attacker_payoffs, attacker_distribution)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RandomCorrelatedGameGenerator):
            return False

        return (
            self.number_of_defender_moves == other.number_of_defender_moves
            and self.number_of_attacker_moves == other.number_of_attacker_moves
            and self.number_of_attacker_types == other.number_of_attacker_types
            and self.correlation_coefficient == other.correlation_coefficient
            and self.uniform_attacker_distribution
            == other.uniform_attacker_distribution
        )
