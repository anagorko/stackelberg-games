"""
This module implements Multiple LPs algorithm from

Conitzer, Vincent, and Tuomas Sandholm.

    "Computing the optimal strategy to commit to."

Proceedings of the 7th ACM conference on Electronic commerce. 2006.

FIXME: write implementation, for normal/bayesian/generalized games; add to testing against dobss.
"""

import itertools
import typing

# noinspection PyPackageRequirements
import pyscipopt  # type: ignore[import-untyped]

from ..games.generalized import GeneralizedBayesianGame


def is_inducible_lp(game: GeneralizedBayesianGame, response: tuple[int, ...]) -> bool:
    """
    Solves a linear problem to check if 'response' is inducible.
    """

    lp = pyscipopt.Model()
    x = {
        i: lp.addVar(f"x[{i}]", vtype="C", lb=-lp.infinity(), ub=lp.infinity())
        for i in range(game.m)
    }
    """Variables that encode defender's mixed strategy."""

    # Add H-representation of game.X to the LP.
    inequalities = game.defender_mixed_strategies.get_inequalities()
    for k in range(inequalities.row_size):
        if k in inequalities.lin_set:
            lp.addCons(
                pyscipopt.quicksum(
                    x[i] * -inequalities[k][i + 1] for i in range(game.m)
                )
                == inequalities[k][0]
            )
        else:
            lp.addCons(
                pyscipopt.quicksum(
                    x[i] * -inequalities[k][i + 1] for i in range(game.m)
                )
                <= inequalities[k][0]
            )

    # Response of each attacker is best by a margin 'delta'

    delta = lp.addVar("delta", vtype="C", lb=-lp.infinity(), ub=lp.infinity())
    for k in range(game.T):
        for j in range(game.n[k]):
            if j != response[k]:
                lp.addCons(
                    pyscipopt.quicksum(
                        x[i] * game.defender_payoffs[k][i][j] for i in range(game.m)
                    )
                    + game.defender_payoffs[k][game.m][j]
                    + delta
                    <= pyscipopt.quicksum(
                        x[i] * game.defender_payoffs[k][i][response[k]]
                        for i in range(game.m)
                    )
                    + game.defender_payoffs[k][game.m][response[k]]
                )
            """Attacker 'k' move 'j' is not better than the chosen move 'response[k]'."""

    lp.setObjective(delta)
    lp.setMaximize()
    lp.hideOutput(True)
    lp.optimize()

    if lp.getStatus() == "optimal":
        return typing.cast(bool, lp.isGT(lp.getVal(delta), 0))

    msg = "is_inducible_lp LP should always have optimal solution."
    raise ValueError(msg)


def inducible_responses_multiple_lps(
    game: GeneralizedBayesianGame,
) -> list[tuple[int, ...]]:
    """
    A brute-force approach to check each attacker's response for inducibility by solving a linear problem.
    """

    return [
        response
        for response in itertools.product(*[range(game.n[t]) for t in range(game.T)])
        if is_inducible_lp(game, response)
    ]
