"""
Experiment runner for cycle graphs.
"""

import argparse
import pprint

import networkx
from termcolor import colored

import stackelberg_games.patrolling as sgp


def main():
    """Runs computations."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=5, type=int, help='Cycle length')
    parser.add_argument('-u', default=1, type=int, help='Number of patrolling units')
    parser.add_argument('-t', default=2, type=int, help='Attack time (uniform)')
    parser.add_argument('-l', default=2, type=int, help='Observation length (attacker)')
    parser.add_argument('-p', default=4, type=int, help='-log of precision')
    args = parser.parse_args()

    environment = sgp.graph_environment(networkx.cycle_graph(args.n), args.u)

    print(colored('Patrolling environment', 'yellow'))
    print()

    pp = pprint.PrettyPrinter(indent=2, compact=True)
    pp.pprint(environment.model_dump())

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

    tau = {t: args.t for t in base.targets}

    """Set up the strategy state and action space."""

    horizon = max(tau.values()) + args.l

    strategy = sgp.TensorProductSpace(
        factors=[
            sgp.PathSpace(
                base_space=base,
                length=horizon
            )
        ]
    )

    projection = {state: state[0][-1] for state in strategy.topology}

    print()
    print(colored('Strategy state and action space', 'yellow'))
    print()
    pp.pprint(strategy.model_dump())

    """Patrolling problem."""

    data = sgp.PatrollingProblem(
        base=base,
        strategy=strategy,
        projection=projection,
        targets=environment.targets,
        tau={t: tau[t] for t in environment.targets},
        reward=environment.reward,
        observation_length=args.l
    )

    solver = sgp.BisectSolver(10**-args.p)
    solution = solver.solve(data)

    solution = solver.mixing_time(solution)
    pp.pprint(solution.model_dump())

    epsilon = 10**-3
    for u in data.strategy.topology:
        for v in solution.transition[u]:
            if solution.transition[u][v] > epsilon and solution.stationary[u] > epsilon:
                print(f'{data.observation[u]} -> {data.projection[v]} {solution.stationary[u]:.3g}: '
                      f'{solution.transition[u][v]:.3g}')


if __name__ == '__main__':
    main()
