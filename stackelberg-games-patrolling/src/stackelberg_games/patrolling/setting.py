"""
This module defines a dataclass that holds information about patrolling setting.
"""

from __future__ import annotations

from fractions import Fraction
import pprint
from typing import Hashable, TypeVar

import networkx
import pydantic


Rational = int | Fraction
Unit = TypeVar('Unit', bound=Hashable)
Location = TypeVar('Location', bound=Hashable)
Route = TypeVar('Route', bound=Hashable)
Target = TypeVar('Target', bound=Hashable)
Coverage = dict[Location, dict[Target, Rational]]


class PatrollingEnvironment(pydantic.BaseModel):
    """A patrolling environment base class."""

    units: tuple[Unit, ...]
    """A set of patrolling units."""
    topology: dict[Unit, networkx.DiGraph]
    """Arbitrary environment topology."""
    length: dict[Unit, dict[Route, int]]
    """Edge lengths."""
    targets: set[Target]
    """A set of targets."""
    coverage: dict[Unit, Coverage]
    """For each unit, protection coverage of each target from each location."""
    reward: dict[Target, Rational]
    """Target weights."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.field_serializer('topology')
    def serialize_topology(self, topology: dict[Unit, networkx.DiGraph]):
        """Serializer for networkx.Graph."""
        return {u: networkx.node_link_data(topology[u]) for u in self.units}


def basilico_et_al() -> PatrollingEnvironment:
    """
    An example from Figure 1 from paper

    Basilico, Nicola, Nicola Gatti, and Francesco Amigoni.
      "Leader-follower strategies for robotic patrolling in environments with arbitrary topologies."
    AAMAS 2009.

    modified to a zero-sum setting: we use attacker's target values.
    """
    units = ('_',)
    topology = {'_': networkx.Graph(['12', '23', '34', '15', '36',
                                     '57', '69', '78', '89', '90']).to_directed()}

    return PatrollingEnvironment(
        units=units,
        topology=topology,
        length={u: {edge: 1 for edge in topology[u].edges} for u in units},
        targets={'4', '5', '0'},
        coverage={u: {v: {w: 1 if w in set(topology[u].neighbors(v)) | {v} else 0
                          for w in topology[u]} for v in topology[u]} for u in units},
        reward={'4': Fraction(2, 10), '5': Fraction(3, 10), '0': Fraction(2, 10)}
    )


def john_et_al(vertices: int = 12, self_loops: bool = False) -> PatrollingEnvironment:
    """
    An example from paper

    John, Yohan, et al.
      "RoSSO: A High-Performance Python Package for Robotic Surveillance Strategy
      Optimization Using JAX."
    arXiv preprint arXiv:2309.08742 (2023).
    """
    d = [[1, 3, 3, 5, 4, 6, 3, 5, 7, 4, 6, 6],
         [3, 1, 5, 4, 2, 4, 4, 5, 5, 3, 5, 5],
         [3, 5, 1, 7, 6, 8, 3, 4, 9, 4, 8, 7],
         [6, 4, 7, 1, 5, 6, 4, 7, 5, 6, 6, 7],
         [4, 3, 6, 5, 1, 3, 5, 5, 6, 3, 4, 4],
         [6, 4, 8, 5, 3, 1, 6, 7, 3, 6, 2, 3],
         [2, 5, 3, 5, 6, 7, 1, 5, 7, 5, 7, 8],
         [3, 5, 2, 7, 6, 7, 3, 1, 9, 3, 7, 5],
         [8, 6, 9, 4, 6, 4, 6, 9, 1, 8, 5, 7],
         [4, 3, 4, 6, 3, 5, 5, 3, 7, 1, 5, 3],
         [6, 4, 8, 6, 4, 2, 6, 6, 4, 5, 1, 3],
         [6, 4, 6, 6, 3, 3, 6, 4, 5, 3, 2, 1]]

    units = (0,)
    topology = {0: networkx.complete_graph(vertices).to_directed()}
    if self_loops:
        for i in range(vertices):
            topology[0].add_edge(i, i)
    targets = set(range(vertices))

    return PatrollingEnvironment(
        units=units,
        topology=topology,
        length={u: {(u, v): d[u][v] for u, v in topology[u].edges} for u in units},
        targets=targets,
        coverage={u: {v: {w: 1 if v == w else 0.0 for w in topology[u]} for v in topology[u]}
                  for u in units},
        reward={t: 1 for t in targets}
    )


def graph_environment(graph: networkx.DiGraph, units: int = 1) -> PatrollingEnvironment:
    """
    Creates a patrolling environment over 'graph' with 'units' units. Edge lengths and rewards
    are set to 1. Coverage is equal to patrol position.
    """
    if not graph.is_directed():
        graph = graph.to_directed()

    units = tuple(range(units))
    topology = {unit: graph for unit in units}
    length = {unit: {(v, w): 1 for v, w in topology[unit].edges} for unit in units}
    targets = set(graph)
    coverage = {u: {v: {w: 1 if v == w else 0 for w in topology[u]}
                    for v in topology[u]} for u in units}
    reward = {t: 1 for t in targets}

    return PatrollingEnvironment(
        units=units,
        topology=topology,
        length=length,
        targets=targets,
        coverage=coverage,
        reward=reward
    )


def main():
    """Used for prototyping."""

    pp = pprint.PrettyPrinter(indent=2, compact=True)

    model = basilico_et_al()
    pp.pprint(model.model_dump())

    model = john_et_al()
    pp.pprint(model.model_dump())

    model = graph_environment(networkx.ladder_graph(4), 1)
    pp.pprint(model.model_dump())


if __name__ == '__main__':
    main()
