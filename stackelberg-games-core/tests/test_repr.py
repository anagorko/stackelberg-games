"""
Unit tests for _repr_html_ methods.
"""

import stackelberg_games.core as sgc  # type: ignore[import-not-found]


def test_repr_html():
    bayesian_game = sgc.RandomIndependentGameGenerator(5, 5, 2).get_instance(0)

    # We only test if code executes without exceptions

    assert bayesian_game._repr_html_() != ""
