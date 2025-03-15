"""
Unit tests for DOBSS implementation.
"""

import pytest

import stackelberg_games.core as sgc  # type: ignore[import-not-found]


def run_on_instance(
    game: sgc.BayesianGame, expected_value: float | None = None
) -> None:
    solver = sgc.DOBSS(game)
    solution = solver.solve()

    if expected_value is not None:
        assert solution.game_value == expected_value

    verbose_model = game.to_dicts()
    assert verbose_model.expected_reward(solution.defender_strategy) == pytest.approx(
        solution.game_value
    )

    normal_form = game.to_normal_form()
    normal_form_solver = sgc.DOBSS(sgc.BayesianGame.from_normal_form(normal_form))
    normal_form_solution = normal_form_solver.solve()

    assert solution.game_value == pytest.approx(normal_form_solution.game_value)


def test_dobss() -> None:
    run_on_instance(sgc.leader_advantage(), 3.0)

    run_on_instance(sgc.RandomIndependentGameGenerator(5, 5, 2).get_instance(0))
