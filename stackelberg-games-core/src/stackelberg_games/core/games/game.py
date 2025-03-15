"""
Game protocols.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pydantic


class Game(Protocol):
    """A protocol implemented by all game classes."""

    def get_metadata(self) -> pydantic.BaseModel:
        """Serializes the Game instance to a pydantic.BaseModel."""

    @classmethod
    def from_metadata(cls, model: pydantic.BaseModel) -> Game:
        """Deserializes a Game instance from a pydantic.BaseModel."""

    def correlation_coefficient(self) -> np.float64:
        """Computes the correlation coefficient of defender and attacker payoff matrices."""

    def to_normal_form(self) -> Game:
        """Converts the Game instance to a normal form."""
