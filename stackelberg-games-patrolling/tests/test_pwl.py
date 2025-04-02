"""
Unit tests for piecewise-linear maps.
"""

from fractions import Fraction

import stackelberg_games.patrolling as sgp


def pair(f, g):
    h = f + g

    cases = {-1, 0, 1, 2, Fraction(1, 3), Fraction(1, 2), Fraction(3, 4)}

    for case in cases:
        assert h(case) == f(case) + g(case)

    h = f.max(g)

    for case in cases:
        assert h(case) == max(f(case), g(case))


def test_pairs():
    f = sgp.PWLMap({0: 0, 1: 1})
    g = sgp.PWLMap({0: 1, 1: 0})

    pair(f, g)

    f = sgp.PWLMap({-1: 0, 1: 1})
    g = sgp.PWLMap({0: 1, 1: 0})

    pair(f, g)
