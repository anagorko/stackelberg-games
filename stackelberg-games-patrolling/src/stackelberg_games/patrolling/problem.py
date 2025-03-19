"""
This module implements a dataclass that holds input data for bisect solver.
"""

import functools
import string

import networkx
import numpy
import pydantic

from .space import State, PatrollingSpace, WeightedGraphSpace, ConcreteSpace, TensorProductSpace, PathSpace, StateActionSpace
from .setting import Rational, PatrollingEnvironment, Target

Observation = tuple[State, ...]
"""In our implementation we restrict observations to be sequences of recently visited states.
This assumption may be easily relaxed."""


def patrolling_problem(environment: PatrollingEnvironment, observation_length: int,
                       tau: dict[Target, int] | int, hidden_states: int = 0):
    """Creates a patrolling problem over a patrolling environment."""

    """Set up the base space."""
    base = PatrollingSpace(
        factors=[
            WeightedGraphSpace(graph=environment.topology[u],
                               d=environment.length[u]) for u in environment.units
        ],
        factor_coverage=[
            environment.coverage[u] for u in environment.units
        ],
        targets=environment.targets,
        reward=environment.reward
    )

    """Set up the attacker."""
    if isinstance(tau, int):
        tau = {t: tau for t in base.targets}

    """Set up the strategy state and action space."""

    horizon = max(tau.values()) + 1 + max(observation_length - 1, 0)

    if hidden_states > 0:
        hidden_states = list(string.ascii_lowercase)[:hidden_states]
        hidden_graph = networkx.DiGraph()
        hidden_graph.add_edges_from(f'{u}{v}' for u in hidden_states for v in hidden_states)
        hidden_state_space = ConcreteSpace(concrete_topology=hidden_graph)

        strategy = TensorProductSpace(
            factors=[
                PathSpace(
                    base_space=base,
                    length=horizon
                ),
                hidden_state_space
            ]
        )
    else:
        strategy = TensorProductSpace(
            factors=[
                PathSpace(
                    base_space=base,
                    length=horizon
                )
            ]
        )

    projection = {state: state[0][-1] for state in strategy.topology}

    return PatrollingProblem(
        base=base,
        strategy=strategy,
        projection=projection,
        targets=environment.targets,
        tau={t: tau[t] for t in environment.targets},
        reward=environment.reward,
        observation_length=observation_length
    )


class PatrollingProblem(pydantic.BaseModel):
    """
    Dataclass holding input for bisect solver.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    base: PatrollingSpace
    """Patrolling state and action space."""
    strategy: StateActionSpace
    """Strategy state and action space."""

    projection: dict[State, State]
    """Projection from strategy space to patrolling space, aka X."""

    targets: set[Target]
    """Set of targets."""

    tau: dict[Target, int]
    """Attack times."""

    observation_length: int
    """Length of attacker's observation. We allow observation_length = 0, i.e. an empty observation."""

    reward: dict[Target, Rational]
    """Target weights."""

    @pydantic.computed_field
    @functools.cached_property
    def observation(self) -> dict[State, Observation]:
        """Observation, aka Y, lifted to the state space."""

        if self.observation_length == 0:
            return {s: () for s in self.strategy.topology}

        obs = {}
        for s in self.strategy.topology:
            # generate a path of length observation_length that ends in s
            path = [s]
            c = s
            for _ in range(self.observation_length - 1):
                c = next(iter(self.strategy.topology.in_edges(c)))[0]
                path.append(c)
            obs[s] = tuple(self.projection[t] for t in path[::-1])

        return obs

    @pydantic.computed_field
    @functools.cached_property
    def actionable(self) -> set[Observation]:
        """Actionable observations."""

        """FIXME: we have a hardcoded assumption here that actionable observations do not end 
        in negative integers. This should be generalized, i.e. there should be a filter function
        specified as an input."""

        if self.observation_length == 0:
            return {()}

        act = set()
        for obs in self.observation.values():
            current_state = obs[-1]
            for f in current_state:
                if not isinstance(f, int) or f >= 0:
                    act.add(obs)
                    break
        return act

    """
    Following fields are needed for formulation of the linear problem. 
    """

    @pydantic.computed_field
    @property
    def memory_x(self) -> int:
        """Length of memory for X."""
        return max(self.tau.values())

    @pydantic.computed_field
    @functools.cached_property
    def history_x(self) -> dict[State, dict[int, State]]:
        """History of X."""

        hx = {}
        for s in self.strategy.topology:
            # generate a path of length memory_x + 1 that ends in s
            path = [s]
            c = s
            for _ in range(self.memory_x):
                c = next(iter(self.strategy.topology.in_edges(c)))[0]
                path.append(c)
            hx[s] = {t: self.projection[p] for t, p in enumerate(path)}

        return hx

    @pydantic.computed_field
    @property
    def memory_y(self) -> int:
        """Length of memory for Y. """
        return max(self.tau.values())

    @pydantic.computed_field
    @functools.cached_property
    def history_y(self) -> dict[Observation, dict[int, State]]:
        """History of Y."""

        hy = {}
        for s in self.strategy.topology:
            # generate a path of length memory_y + 1 that ends in s
            path = [s]
            c = s
            for _ in range(self.memory_y):
                c = next(iter(self.strategy.topology.in_edges(c)))[0]
                path.append(c)
            hy[s] = {t: self.observation[p] for t, p in enumerate(path)}
        return hy

    @pydantic.computed_field
    @functools.cached_property
    def capture_probability(self) -> dict[State, dict[Target, Rational]]:
        """Capture probability, D(s, j) from the paper."""

        d = {}

        for s in self.strategy.topology:
            d[s] = {}
            for j in self.targets:
                d[s][j] = 1 - numpy.prod([1 - self.base.coverage[self.history_x[s][t]][j]
                                          for t in range(self.tau[j] + 1)])
        return d
