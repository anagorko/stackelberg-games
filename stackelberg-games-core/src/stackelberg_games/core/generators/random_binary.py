"""
This module implements generators for Bayesian games with random binary payoff matrices.
"""

import numpy as np

from ..games.bayesian import BayesianGame
from .generator import GameGenerator, GameGeneratorMetadata


class RandomBinaryGameGenerator(GameGenerator):
    """
    Independent payoff matrices with payoffs sampled with uniform probability from { 0, 1 }.
    """

    tag: str = "random_independent"

    def __init__(
        self,
        number_of_defender_moves: int,
        number_of_attacker_moves: int,
        number_of_attacker_types: int,
        uniform_attacker_distribution: bool = False,
    ) -> None:
        self.number_of_defender_moves = number_of_defender_moves
        self.number_of_attacker_moves = number_of_attacker_moves
        self.number_of_attacker_types = number_of_attacker_types
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
                "uniform_attacker_distribution": self.uniform_attacker_distribution,
            },
        )

    def get_instance(self, random_seed: int) -> BayesianGame:
        """"""
        rng = np.random.default_rng(random_seed)

        defender_payoffs = [
            rng.choice(
                [-1, 1],
                size=(self.number_of_defender_moves, self.number_of_attacker_moves),
            )
            for _ in range(self.number_of_attacker_types)
        ]
        attacker_payoffs = [
            rng.choice(
                [-1, 1],
                size=(self.number_of_defender_moves, self.number_of_attacker_moves),
            )
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
        if not isinstance(other, RandomBinaryGameGenerator):
            return False

        return (
            self.number_of_defender_moves == other.number_of_defender_moves
            and self.number_of_attacker_moves == other.number_of_attacker_moves
            and self.number_of_attacker_types == other.number_of_attacker_types
            and self.uniform_attacker_distribution
            == other.uniform_attacker_distribution
        )
