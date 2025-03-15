"""
Unit tests for implementation of Bayesian Stackelberg games.
"""

import pytest

import numpy as np

import stackelberg_games.core as sgc  # type: ignore[import-not-found]


def abr_single_instance(game: sgc.BayesianGame) -> None:
    solver = sgc.DOBSS(game)
    solution = solver.solve()

    defender_strategy = np.array(
        list(solution.defender_strategy.values()), dtype=np.float64
    )
    abr = game.attacker_best_response(defender_strategy)

    value = sum(
        game.attacker_distribution[t]
        * defender_strategy[i]
        * game.defender_payoffs[t][i, abr[t]]
        for i in range(game.m)
        for t in range(game.T)
    )

    assert value == pytest.approx(solution.game_value)


def test_abr() -> None:
    abr_single_instance(sgc.RandomIndependentGameGenerator(5, 5, 2).get_instance(0))
    abr_single_instance(sgc.RandomIndependentGameGenerator(4, 4, 4).get_instance(0))
    abr_single_instance(sgc.RandomIndependentGameGenerator(6, 6, 6).get_instance(0))
