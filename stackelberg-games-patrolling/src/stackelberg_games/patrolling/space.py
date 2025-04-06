"""
This module implements dataclasses that represent state and action spaces.
"""

import abc
from fractions import Fraction
import functools
import itertools
import random
from typing import Hashable, TypeVar

import networkx
import numpy
import pydantic

from .setting import Coverage, Target

State = TypeVar('State', bound=Hashable)
Action = tuple[State, State]


class StateActionSpace(pydantic.BaseModel, abc.ABC):
    """
    A base class for state and action spaces.

    The state and action space represented by this object is accessible via .topology property.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.field_serializer('topology')
    def serialize_graph(self, topology: networkx.DiGraph):
        """Serializer for networkx.Graph."""
        return networkx.node_link_data(topology)

    @pydantic.computed_field
    @abc.abstractmethod
    @functools.cached_property
    def topology(self) -> networkx.DiGraph:
        """The state and action space."""


class PatrollingStateActionSpace(StateActionSpace):
    """
    A base class for state and action spaces with coverage functions.
    """

    targets: set[Target]
    """Coverage targets."""

    @pydantic.computed_field
    @abc.abstractmethod
    @functools.cached_property
    def coverage(self) -> dict[State, dict[Target, float]]:
        """Target coverage for each state."""


class WeightedGraphSpace(StateActionSpace):
    """An action-state space created by expanding long edges of a weighted graph."""

    graph: networkx.DiGraph
    """Input graph."""
    d: dict[State, int]
    """Edge lengths."""

    @pydantic.computed_field
    @functools.cached_property
    def topology(self) -> networkx.DiGraph:
        """Segments long edges into multiple edges of unit length.
        """

        assert (all(not isinstance(v, int) or v >= 0 for v in self.graph))

        def new_nodes(g, n):
            nodes = []
            for _ in range(n):
                while new_nodes.n in g:
                    new_nodes.n -= 1
                nodes.append(new_nodes.n)
                new_nodes.n -= 1
            return nodes

        new_nodes.n = -1

        unrolled = self.graph.copy()
        assert (all(isinstance(v, int) for v in self.d.values()))

        target = {}
        pos = {}
        for u, v in self.d:
            if self.d[u, v] > 1:
                nn = new_nodes(unrolled, self.d[u, v] - 1)
                for i, w in enumerate(nn):
                    target[w] = v
                    t = Fraction(i+1, len(nn) + 1)
                    v_pos = self.graph.nodes[v]["pos"]
                    u_pos = self.graph.nodes[u]["pos"]
                    pos[w] = [t * v_pos[0] + (1-t) * u_pos[0], t * v_pos[1] + (1-t) * u_pos[1]]
                path = [u] + nn + [v]
                unrolled.add_edges_from(list(zip(path, path[1:])))
                unrolled.remove_edge(u, v)
        networkx.set_node_attributes(unrolled, target, 'target')
        networkx.set_node_attributes(unrolled, pos, 'pos')
        return unrolled


class PathSpace(StateActionSpace):
    """An action-state space generated from paths of specified length built on a graph."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    base_space: StateActionSpace
    """Underlying action-state space."""
    length: int
    """Length of paths."""

    @pydantic.computed_field
    @functools.cached_property
    def topology(self) -> networkx.DiGraph:
        def find_paths(u, n: int):
            if n == 1:
                return [[u]]
            return [[u] + path for neighbor in self.base_space.topology.neighbors(u)
                    for path in find_paths(neighbor, n - 1)]

        states = {tuple(path) for v in self.base_space.topology
                  for path in find_paths(v, self.length)}

        def target(v):
            dist = 0
            while isinstance(v, int) and v < 0:
                v = next(iter(self.base_space.topology.out_edges(v)))[1]
                dist += 1
            return v, dist

        actions = {
            (state, state[1:] + (v,)) for state in states
            for v in self.base_space.topology.neighbors(state[-1])
        }

        graph = networkx.DiGraph()
        graph.add_nodes_from(states)
        graph.add_edges_from(actions)

        node_labels = {v: ''.join(self.base_label[u] for u in v) for v in graph}
        networkx.set_node_attributes(graph, node_labels, name='label')
        edge_labels = {
            (v, w): self.base_label[target(w[-1])[0]] for v, w in graph.edges
        }
        networkx.set_edge_attributes(graph, edge_labels, name='label')
        return graph

    @pydantic.computed_field
    @functools.cached_property
    def base_label(self) -> dict:
        """Node names, with intermediate node names replaced with '-' for readability."""
        return {v: '-' if isinstance(v, int) and v < 0 else str(v)
                for v in self.base_space.topology}

    @pydantic.computed_field
    @functools.cached_property
    def target_node(self) -> dict:
        """
        Target node for intermediate nodes.
        """

        target = {}
        for w in self.topology:
            v = w[-1]
            while isinstance(v, int) and v < 0:
                _, v = next(iter(self.base_space.topology.out_edges(v)))
            target[w] = v
        return target

    def graph_representation(self):
        """
        A graph representation with node and edge labels for rendering.
        """
        g = networkx.DiGraph()
        g.add_nodes_from(self.state)
        g.add_edges_from([(u, v) for u in self.state for v in self.action[u]])
        networkx.set_node_attributes(g, {s: self.label[s] for s in self.state}, name='label')

        networkx.set_edge_attributes(g, {(u, v): self.target_node[v] for u in self.state
                                         for v in self.action[u]}, name='label')
        return g


class ProjectedSpace(StateActionSpace):
    """A projection of a state and action space."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    base_space: StateActionSpace
    """Underlying action-state space."""
    projection: dict[State, State]
    """A projection."""

    @pydantic.computed_field
    @functools.cached_property
    def topology(self) -> networkx.DiGraph:
        g = networkx.DiGraph()
        g.add_nodes_from([self.projection[v] for v in self.base_space.topology])
        g.add_edges_from([(self.projection[u], self.projection[v])
                          for u, v in self.base_space.topology.edges])
        return g


class SubSpace(StateActionSpace):
    """A subspace of a state and action space."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    base_space: StateActionSpace
    """Underlying action-state space."""
    subset: set[State]
    """A projection."""

    @pydantic.computed_field
    @functools.cached_property
    def topology(self) -> networkx.DiGraph:
        g = networkx.DiGraph()
        g.add_nodes_from(self.subset)
        g.add_edges_from([(u, v) for u, v in self.base_space.topology.edges
                          if u in self.subset and v in self.subset])
        return g


class ConcreteSpace(StateActionSpace):
    """A state and action space with explicitly given underlying digraph."""

    concrete_topology: networkx.DiGraph

    @pydantic.computed_field
    @functools.cached_property
    def topology(self) -> networkx.DiGraph:
        return self.concrete_topology


class TensorProductSpace(StateActionSpace):
    """A tensor product state and action space."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    factors: list[StateActionSpace]
    """Factor spaces."""

    @pydantic.computed_field
    @functools.cached_property
    def topology(self) -> networkx.DiGraph:
        vertices = list(itertools.product(*[factor.topology.nodes for factor in self.factors]))

        graph = networkx.DiGraph()
        graph.add_nodes_from(vertices)

        for u in graph:
            targets = itertools.product(*[f.topology.neighbors(v) for v, f in zip(u, self.factors)])
            graph.add_edges_from([(u, t) for t in targets])

        return graph


class PatrollingSpace(TensorProductSpace):
    """A tensor product state and action space with a coverage function."""

    factor_coverage: list[Coverage]
    targets: set[Target]
    reward: dict[Target, float]

    @pydantic.computed_field
    @functools.cached_property
    def coverage(self) -> Coverage:
        """
        This is a map that assigns to product states maps from targets to protection probabilities.
        """
        null_coverage = {t: 0 for t in self.targets}

        c = {}
        for v in self.topology:
            c[v] = {}
            for t in self.targets:
                c[v][t] = 1 - numpy.prod([1 - coverage.get(location, null_coverage)[t]
                                          for location, coverage in zip(v, self.factor_coverage)])
        return c


def main():
    """Used for prototyping."""
    random.seed(0)

    g = networkx.DiGraph()
    g.add_edges_from(['ab', 'aa', 'ba', 'bb'])
    d = {e: random.randint(1, 3) for e in g.edges}
    base = WeightedGraphSpace(graph=g, d=d)

    print(base.topology.nodes)
    print(base.topology.edges)
    print(TensorProductSpace(factors=[base, base, base]).topology)


if __name__ == '__main__':
    main()
