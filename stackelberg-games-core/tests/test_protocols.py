"""
This module implements unit tests for protocol implementation.

Note that the actual checks are done statically by mypy in pre-commit, not by pytest at runtime.

TODO: find a way to remove type: ignore below
"""

from typing import TYPE_CHECKING

import numpy as np

from stackelberg_games import core  # type: ignore[import-not-found]


def test_game_protocol() -> None:
    m, n = 7, 12

    rng = np.random.default_rng()

    defender_payoff = rng.random(size=(m, n))
    attacker_payoff = rng.random(size=(m, n))

    if TYPE_CHECKING:
        game: core.Game = core.NormalFormGame(defender_payoff, attacker_payoff)  # noqa: F841
