"""
This module implements our linear program + binary search solver.
"""

from fractions import Fraction
import time

import pyscipopt
from termcolor import colored

from .plan import DefensePlan
from .problem import PatrollingProblem
from .setting import Rational
from .solver import Solver


class BisectSolver(Solver):
    def __init__(self, precision: float = 10 ** -2, verbose=True):
        self.precision = precision
        self.verbose = verbose

    def mixing_time(self, solution: DefensePlan):
        model = pyscipopt.Model()

        t = {(i, j): model.addVar(f't[{i}, {j}]', vtype='C', lb=0, ub=1)
             for i, j in solution.data.strategy.topology.edges}

        # Network-flow constraints
        for i in solution.data.strategy.topology:
            model.addCons(pyscipopt.quicksum(t[i, j] for j in solution.data.strategy.topology.neighbors(i))
                          == solution.stationary[i])
            model.addCons(pyscipopt.quicksum(t[j, i] for j, _ in solution.data.strategy.topology.in_edges(i))
                          == solution.stationary[i])

        m = {(i, j): model.addVar(f'm[{i}, {j}]', vtype='C', lb=0, ub=model.infinity())
             for i, j in solution.data.strategy.topology.edges}
        mu = {i: model.addVar(f'mu[{i}]', vtype='C', lb=0, ub=1) for i in solution.data.strategy.topology}

        for i, j in solution.data.strategy.topology.edges:
            model.addCons(m[i, j] >= mu[i] - t[i, j])
            model.addCons(m[i, j] >= t[i, j] - mu[i])

        model.setObjective(pyscipopt.quicksum(m[i, j] for i, j in solution.data.strategy.topology.edges))
        model.setMinimize()
        if not self.verbose:
            model.hideOutput()
        model.optimize()

        if model.getStatus() == 'optimal':
            optimal_solution = {
                'status': 'optimal',
                'p': {v: solution.stationary[v] for v in solution.data.strategy.topology},
                't': {u: {v: 0 if model.isZero(solution.stationary[u]) else model.getVal(t[u, v]) /
                                                                            solution.stationary[u]
                          for _, v in solution.data.strategy.topology.out_edges(u)}
                      for u in solution.data.strategy.topology}
            }
        else:
            optimal_solution = {'status': model.getStatus()}

        if model.getStatus() == 'optimal':
            return DefensePlan(
                data=solution.data,
                stationary=optimal_solution['p'],
                transition=optimal_solution['t'],
                lower_bound=solution.lower_bound,
                upper_bound=solution.upper_bound
            )

    def linear_program(self, xi: float, data: PatrollingProblem):
        model = pyscipopt.Model()

        num = 1

        model.setParam('numerics/epsilon', 1e-10)
        model.setParam('numerics/dualfeastol', 1e-10)
        model.setParam('numerics/feastol', 1e-10)

        s = {i: model.addVar(f's[{i}', vtype='C', lb=0, ub=num) for i in data.strategy.topology}
        t = {(i, j): model.addVar(f't[{i}, {j}]', vtype='C', lb=0, ub=num) for i, j in data.strategy.topology.edges}

        # Network-flow constraints
        model.addCons(pyscipopt.quicksum(s[i] for i in data.strategy.topology) == num)
        for i in data.strategy.topology:
            model.addCons(pyscipopt.quicksum(t[i, j] for _, j in data.strategy.topology.out_edges(i)) == s[i])
            model.addCons(pyscipopt.quicksum(t[j, i] for j, _ in data.strategy.topology.in_edges(i)) == s[i])

        for i in data.actionable:
            for j in data.targets:
                states = [state for state in data.strategy.topology if data.history_y[state][data.tau[j]] == i]
                model.addCons(pyscipopt.quicksum(s[state] * (data.reward[j] * data.capture_probability[state][j]
                                                             - xi) for state in states) >= 0)

        model.setObjective(0)
        model.hideOutput()
        model.optimize()

        if model.getStatus() != 'optimal':
            return {
                'status': model.getStatus(),
            }

        solution = {
            'p': {
                v: 0 if model.isZero(model.getVal(s[v])) else model.getVal(s[v]) / num
                for v in data.strategy.topology
            },
            't': {
                u: {
                    v: 1 / data.strategy.topology.out_degree(u) if model.isZero(model.getVal(s[u])) else
                    model.getVal(t[u, v]) / model.getVal(s[u]) for _, v in data.strategy.topology.out_edges(u)}
                for u in data.strategy.topology
            }
        }
        return {'status': 'optimal'} | solution

    def solve(self, data: PatrollingProblem, lb: Rational = None, ub: Rational = None):
        if lb is None:
            lower_bound = Fraction(0, 1)
        else:
            lower_bound = lb

        if ub is None:
            upper_bound = Fraction(max(data.reward.values()))
        else:
            upper_bound = ub

        optimal_solution = None

        if self.verbose:
            print()
            print(colored('Solving...', 'yellow'))
            print()

        start_time = time.time()
        while upper_bound - lower_bound > self.precision:
            midpoint = (lower_bound + upper_bound) / 2
            solution = self.linear_program(midpoint, data)
            end_time = time.time()

            if solution['status'] == 'optimal':
                optimal_solution = solution
                lower_bound = midpoint
            else:
                upper_bound = midpoint

            if self.verbose:
                print(f'Found bounds {float(lower_bound):.5g} ... {float(upper_bound):.5g}, '
                      f'elapsed time {end_time - start_time:.3g}s')

        if self.verbose:
            print()
            print(colored('Optimal value in range ', 'blue'), f'({lower_bound}, {upper_bound})')
            print()

        if optimal_solution is None:
            return DefensePlan(
                data=data,
                # Uniform distributions for the purpose of Monte Carlo validation
                stationary={v: 1./data.strategy.topology.number_of_nodes() for v in data.strategy.topology},
                transition={u: {v: 1./len(data.strategy.topology.out_edges(u)) for _, v in data.strategy.topology.out_edges(u)} for u in data.strategy.topology},
                lower_bound=0,
                upper_bound=0
            )

        return DefensePlan(
            data=data,
            stationary=optimal_solution['p'],
            transition=optimal_solution['t'],
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
