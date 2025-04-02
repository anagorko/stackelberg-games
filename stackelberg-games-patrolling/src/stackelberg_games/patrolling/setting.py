"""
This module defines a dataclass that holds information about patrolling setting.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Hashable, TypeVar

import geopy.distance  # type: ignore[import-untyped]
import networkx as nx
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
    topology: dict[Unit, nx.DiGraph]
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
    def serialize_topology(self, topology: dict[Unit, nx.DiGraph]):
        """Serializer for nx.Graph."""
        return {u: nx.node_link_data(topology[u]) for u in self.units}


def basilico_et_al() -> PatrollingEnvironment:
    """
    An example from Figure 1 from paper

    Basilico, Nicola, Nicola Gatti, and Francesco Amigoni.
      "Leader-follower strategies for robotic patrolling in environments with arbitrary topologies."
    AAMAS 2009.

    modified to a zero-sum setting: we use attacker's target values.
    """
    units = ('_',)
    topology = {'_': nx.Graph(['12', '23', '34', '15', '36',
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
    topology = {0: nx.complete_graph(vertices).to_directed()}
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


def graph_environment(graph: nx.DiGraph, units: int = 1) -> PatrollingEnvironment:
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


def gdynia_graph(edge_len_unit: int = 1000) -> nx.DiGraph[str]:
    """
    The running example topology from the UAI'24 paper.
    """

    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(
        [
            (
                "Aw_in1",
                {"pos": (54.538890756145584, 18.56115149288245)},
            ),  # Awanport --- wejście 1
            (
                "Aw_in2",
                {"pos": (54.536077259063106, 18.561323154247752)},
            ),  # Awanport --- wejście 2
            (
                "B_x_xi",
                {"pos": (54.538890756145584, 18.554456699635608)},
            ),  # Baseny X oraz XI
            (
                "Aw_out",
                {"pos": (54.53533768302353, 18.547958199361737)},
            ),  # # Awanport --- wyjście
            ("B_ix", {"pos": (54.53638344920011, 18.54087716804296)}),  # Basen IX
            ("B_viii", {"pos": (54.54076099379705, 18.52120498589296)}),  # Basen VIII
            ("B_vii", {"pos": (54.539441473596035, 18.521333731916936)}),  # Basen VII
            ("B_vi", {"pos": (54.53778578884019, 18.527642287091847)}),  # Basen VI
            ("B_v", {"pos": (54.536204733934866, 18.533736265560126)}),  # Basen V
            ("B_iv", {"pos": (54.53390518408724, 18.540637593293866)}),  # Basen IV
        ]
    )
    g.add_edge("Aw_in1", "Aw_in2")
    g.add_edge("B_x_xi", "Aw_in1")
    g.add_edge("B_x_xi", "Aw_in2")
    g.add_edge("Aw_out", "Aw_in1")
    g.add_edge("Aw_out", "Aw_in2")
    g.add_edge("Aw_out", "B_x_xi")
    g.add_edge("B_ix", "Aw_out")
    g.add_edge("B_iv", "Aw_out")
    g.add_edge("B_iv", "B_ix")
    g.add_edge("B_v", "B_ix")
    g.add_edge("B_v", "B_iv")
    g.add_edge("B_vi", "B_v")
    g.add_edge("B_vii", "B_vi")
    g.add_edge("B_viii", "B_vi")
    g.add_edge("B_viii", "B_vii")

    # Setting length of the edges in meters and in abstract units
    def dist(e: tuple[str, str]) -> float:
        return geopy.distance.geodesic(g.nodes[e[0]]["pos"], g.nodes[e[1]]["pos"]).m

    # noinspection PyTypeChecker
    nx.set_edge_attributes(
        g,
        {e: max(1, round(dist(e) / edge_len_unit)) for e in g.edges},
        "len",
    )

    nx.set_edge_attributes(
        g,
        {e: int(dist(e)) for e in g.edges},
        "dist",
    )

    return g.to_directed()


def port_gdynia(number_of_units: int = 1) -> PatrollingEnvironment:
    """
    The running example setting from the UAI'24 paper.
    """
    layout = gdynia_graph()

    units = tuple(range(number_of_units))
    topology = {
        unit: layout for unit in units
    }
    length = {
        unit: {e: layout.edges[e]["len"] for e in layout.edges}
        for unit in units
    }
    targets = {
        v for v in layout if v.startswith("B")
    }  # The docks are the targets

    coverage = {
        u: {
            v: {
                w:
                    1
                    if v == w  # Coverage when exactly at the node
                    else Fraction(1, 2)
                    if w in topology[u].neighbors(v)  # Coverage when at a neighbor
                    else 0
                for w in topology[u]
            }  # Coverage when in other nodes
            for v in topology[u]
        }
        for u in units
    }
    reward = {t: 1 for t in targets}

    return PatrollingEnvironment(
        units=units,
        topology=topology,
        length=length,
        targets=targets,
        coverage=coverage,
        reward=reward,
    )
