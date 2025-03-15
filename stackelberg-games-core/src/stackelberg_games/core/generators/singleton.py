"""
This module implements generators for some interesting singleton game instances.
"""

from ..games.bayesian import BayesianGame


def leader_advantage() -> BayesianGame:
    """
    A very simple game from the introduction of

    Paruchuri, Praveen, et al.
      "Playing games for security: An efficient exact algorithm for solving Bayesian Stackelberg games."
    Proceedings of the 7th international joint conference on Autonomous agents and multiagent systems-Volume 2. 2008.

    which demonstrates that the defender may have an advantage from disclosing his strategy to the attacker.
    """

    X: set[str | int] = {"a", "b"}
    T: set[str | int] = {1, 2}
    J: dict[str | int, set[str | int]] = {1: {"c", "d"}, 2: {"c'", "d'"}}
    R: dict[str | int, dict[str | int, dict[str | int, float]]] = {
        1: {"a": {"c": 2, "d": 4}, "b": {"c": 1, "d": 3}},
        2: {"a": {"c'": 1, "d'": 2}, "b": {"c'": 0, "d'": 3}},
    }
    C: dict[str | int, dict[str | int, dict[str | int, float]]] = {
        1: {"a": {"c": 1, "d": 0}, "b": {"c": 0, "d": 2}},
        2: {"a": {"c'": 1, "d'": 0}, "b": {"c'": 1, "d'": 2}},
    }
    P: dict[str | int, float] = {1: 0.5, 2: 0.5}

    return BayesianGame.from_dicts(X, T, J, R, C, P)
