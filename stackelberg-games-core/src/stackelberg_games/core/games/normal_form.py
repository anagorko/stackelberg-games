"""
This module implements a data class for games in a normal form.
"""

from __future__ import annotations
from typing import cast, Sequence

import numpy as np
import numpy.typing as npt
import pydantic

from .game import Game
from ..linalg import to_matrix


class NormalFormGame(Game):
    """
    A game in a normal form.
    """

    def __init__(
        self,
        defender_payoff: npt.ArrayLike,
        attacker_payoff: npt.ArrayLike,
        defender_moves: Sequence[str | int] | None = None,
        attacker_moves: Sequence[str | int] | None = None,
        defender_name: str | int | None = None,
        attacker_name: str | int | None = None,
    ) -> None:
        self.defender_payoff: npt.NDArray[np.float64] = to_matrix(defender_payoff)
        self.attacker_payoff: npt.NDArray[np.float64] = to_matrix(attacker_payoff)

        assert len(self.defender_payoff.shape) == 2
        assert self.defender_payoff.shape == self.attacker_payoff.shape

        if defender_moves is None:
            self.defender_moves: tuple[str | int, ...] = tuple(
                range(self.defender_payoff.shape[0])
            )
        else:
            self.defender_moves = tuple(defender_moves)

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

    class NormalFormDataclass(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

        defender_payoff: list[list[float]]
        attacker_payoff: list[list[float]]
        defender_moves: tuple[str | int, ...]
        attacker_moves: tuple[str | int, ...]
        defender_name: str | int
        attacker_name: str | int

    def get_metadata(self) -> NormalFormGame.NormalFormDataclass:
        return NormalFormGame.NormalFormDataclass(
            defender_payoff=self.defender_payoff.tolist(),
            attacker_payoff=self.attacker_payoff.tolist(),
            defender_moves=self.defender_moves,
            attacker_moves=self.attacker_moves,
            defender_name=self.defender_name,
            attacker_name=self.attacker_name,
        )

    @classmethod
    def from_metadata(cls, model: pydantic.BaseModel) -> NormalFormGame:
        assert isinstance(model, NormalFormGame.NormalFormDataclass)

        return NormalFormGame(
            np.array(model.defender_payoff, dtype=np.float64),
            np.array(model.attacker_payoff, dtype=np.float64),
            model.defender_moves,
            model.attacker_moves,
            model.defender_name,
            model.attacker_name,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormalFormGame):
            return False

        return (
            np.array_equal(self.defender_payoff, other.defender_payoff)
            and np.array_equal(self.attacker_payoff, other.attacker_payoff)
            and self.defender_moves == other.defender_moves
            and self.attacker_moves == other.attacker_moves
            and self.defender_name == other.defender_name
            and self.attacker_name == other.attacker_name
        )

    def correlation_coefficient(self) -> np.float64:
        """Computes a correlation coefficient between defender and attacker payoff matrices."""

        if (
            np.linalg.norm(self.defender_payoff) == 0
            or np.linalg.norm(self.attacker_payoff) == 0
        ):
            return np.float64(0)

        return cast(
            np.float64,
            np.linalg.tensordot(self.defender_payoff, self.attacker_payoff)
            / (
                np.linalg.norm(self.defender_payoff)
                * np.linalg.norm(self.attacker_payoff)
            ),
        )

    def to_normal_form(self) -> NormalFormGame:
        return self
