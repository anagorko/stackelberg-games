"""
This module implements generators for Bayesian games with circulant payoff matrices.
"""

import numpy as np
import scipy  # type: ignore[import-untyped]

from ..games.bayesian import BayesianGame
from .generator import GameGenerator, GameGeneratorMetadata


class CirculantGameGenerator(GameGenerator):
    """
    Circulant payoff matrices with first columns sampled from the standard normal distribution.
    """

    tag: str = "circulant"

    def __init__(
        self,
        number_of_moves: int,
        number_of_attacker_types: int,
        uniform_attacker_distribution: bool = False,
    ) -> None:
        self.number_of_moves = number_of_moves
        self.number_of_attacker_types = number_of_attacker_types
        self.uniform_attacker_distribution = uniform_attacker_distribution

    def get_metadata(self) -> GameGeneratorMetadata:
        """Serializes the Generator instance to a pydantic.BaseModel."""
        return GameGeneratorMetadata(
            cls=self.__class__,
            args=(),
            kwargs={
                "number_of_moves": self.number_of_moves,
                "number_of_attacker_types": self.number_of_attacker_types,
                "uniform_attacker_distribution": self.uniform_attacker_distribution,
            },
        )

    def get_instance(self, random_seed: int) -> BayesianGame:
        """"""
        rng = np.random.default_rng(random_seed)

        defender_payoffs = [
            scipy.linalg.circulant(rng.normal(size=self.number_of_moves))
            for _ in range(self.number_of_attacker_types)
        ]
        attacker_payoffs = [
            scipy.linalg.circulant(rng.normal(size=self.number_of_moves))
            for _ in range(self.number_of_attacker_types)
        ]

        if self.uniform_attacker_distribution:
            attacker_distribution: list[float] = [
                1 / self.number_of_attacker_types
            ] * self.number_of_attacker_types
        else:
            attacker_distribution = rng.dirichlet(
                (1,) * self.number_of_attacker_types
            ).tolist()

        return BayesianGame(defender_payoffs, attacker_payoffs, attacker_distribution)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CirculantGameGenerator):
            return False

        return (
            self.number_of_moves == other.number_of_moves
            and self.number_of_attacker_types == other.number_of_attacker_types
            and self.uniform_attacker_distribution
            == other.uniform_attacker_distribution
        )
