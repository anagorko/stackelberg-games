"""
This module implements a base class for solvers.
"""

import abc

from .problem import PatrollingProblem
from .plan import DefensePlan
from .setting import Rational


class Solver(abc.ABC):
    """A base class for solvers."""

    @abc.abstractmethod
    def solve(self, data: PatrollingProblem, lb: Rational = None, ub: Rational = None) -> DefensePlan:
        """Solves a patrolling problem."""
