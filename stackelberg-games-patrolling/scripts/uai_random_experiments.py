import functools
import string
import time
from typing import Hashable, TypeVar, Callable

import networkx
import numpy
import pydantic
import pyscipopt

import csv
import datetime
import os
import random
import pandas
import seaborn

import stackelberg_games.patrolling as sgp


def run_series(graph_name: str, graph_generator: Callable[[],networkx.DiGraph]):
    
    solver = sgp.BisectSolver(precision=10**-3, verbose=False)
    rollout_len = 10**3
    rollouts_num = 10**3
    
    out_dir_path = f"{sgp.directories.results_dir}/random_results/"
    os.makedirs(out_dir_path, exist_ok=True)
    with open(f'{out_dir_path}{graph_name}_{round(datetime.datetime.timestamp(datetime.datetime.now()))}.csv', 'w', encoding='utf-8', newline='') as outputFile:
        out = csv.writer(outputFile, delimiter=';')

        def output_row(row):
            out.writerow(row)
            print(row)
        
        output_row([graph_name])
        output_row(['graph_size', 'attack_time', 'observation', 'stat', 'value'])
        for n in range(3, 11):
            g = graph_generator(n)
            environment = sgp.graph_environment(g, units=1)
            for attack_time in range(2, 4):
                tau = { v : attack_time for v in g }
                for observation in range(0, 3):
                    data = sgp.patrolling_problem(environment, observation, tau)
                    start_time = time.time()
                    solution = solver.solve(data)
                    end_time = time.time()
                    
                    row_prefix = [n, attack_time, observation]
                    output_row(row_prefix + ['lp_lower_bound', float(solution.lower_bound)])
                    output_row(row_prefix + ['lp_upper_bound', float(solution.upper_bound)])
                    output_row(row_prefix + ['lp_runtime', end_time - start_time])


def main():
    random.seed(42)

    for sample in range(100):
        ba2_gen = lambda size: networkx.barabasi_albert_graph(size, 2)
        run_series('ba2', ba2_gen)
        
        er2_gen = lambda size: networkx.erdos_renyi_graph(size, 2./(size-1))
        run_series('er2', er2_gen)
        
        ws2_gen = lambda size: networkx.watts_strogatz_graph(size, 2, .25)
        run_series('ws2', ws2_gen)

if __name__ == '__main__':
    main()