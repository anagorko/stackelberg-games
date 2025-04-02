"""
Experiment runner for star graphs.
"""

import pprint
import string

import networkx
from termcolor import colored

import stackelberg_games.patrolling as sgp


def main():
    """Used for prototyping."""

    """A star graph with long arms."""
    u = 1  # number of patrolling units
    n = 5  # number of arms
    m = 1  # length of arms
    # attack_time = 2*m + 1  # attack time
    attack_time = 2
    observation_length = 1

    star = networkx.Graph()
    leaves = list(string.ascii_lowercase)[:n]
    star.add_edges_from(f's{v}' for v in leaves)
    star = star.to_directed()

    """Set up patrolling environment."""
    units = tuple(range(u))
    topology = {unit: star for unit in units}
    length = {unit: {(v, w): m for v, w in topology[unit].edges} for unit in units}
    targets = {v for v in leaves}
    coverage = {u: {v: {w: 1.0 if v == w else 0.0 for w in topology[u]} for v in topology[u]} for u in units}
    reward = {t: 1 for t in targets}

    environment = sgp.PatrollingEnvironment(
        units=units,
        topology=topology,
        length=length,
        targets=targets,
        coverage=coverage,
        reward=reward
    )

    print(colored('Patrolling environment', 'yellow'))
    print()

    pp = pprint.PrettyPrinter(indent=2, compact=True)
    pp.pprint(environment.model_dump())

    """Set up the patrolling state and acton space."""
    base = sgp.PatrollingSpace(
        factors=[
            sgp.WeightedGraphSpace(graph=environment.topology[u],
                                   d=environment.length[u]) for u in environment.units
        ],
        factor_coverage=[
            environment.coverage[u] for u in environment.units
        ],
        targets=environment.targets,
        reward=environment.reward
    )

    print()
    print(colored('Patrolling state and action space', 'yellow'))
    print()
    pp.pprint(base.model_dump())

    """Set up the attacker."""

    tau = {t: attack_time for t in base.targets}
    # cost = base.reward

    """Set up the strategy state and action space."""

    horizon = max(tau.values()) + observation_length

    hidden_graph = networkx.DiGraph()
    hidden_graph.add_edges_from(['xx', 'xy', 'yx', 'yy'])
    hidden_states = sgp.ConcreteSpace(concrete_topology=hidden_graph)

    strategy = sgp.TensorProductSpace(
        factors=[
            sgp.PathSpace(
                base_space=base,
                length=horizon
            ),
            hidden_states,
        ]
    )

    # print()
    # print(colored('Strategy state and action space', 'yellow'))
    # print()
    # pp.pprint(strategy.model_dump())

    projection = {state: state[0][-1] for state in strategy.topology}

    """Patrolling problem."""

    data = sgp.PatrollingProblem(
        base=base,
        strategy=strategy,
        projection=projection,
        targets=environment.targets,
        tau={t: tau[t] for t in environment.targets},
        reward=environment.reward,
        observation_length=observation_length
    )

    print()
    print(colored('Problem parameters', attrs=['bold']))
    print()
    print(f'Units: {u}.\n'
          f'Arms: {n}.\n'
          f'Arm length: {m}.\n'
          f'Attack time: {attack_time}.\n'
          f'Observation length: {observation_length}.\n'
          f'Actionable observations: {data.actionable}.\n'
          f'Number of nodes in strategy space: {len(data.strategy.topology.nodes)}\n')

    solver = sgp.BisectSolver(10**-2)
    solution = solver.solve(data)
    pp.pprint(solution.model_dump())

    pp.pprint(set(data.observation.values()))


if __name__ == '__main__':
    main()
