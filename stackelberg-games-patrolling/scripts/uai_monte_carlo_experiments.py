"""
Comparing Monte Carlo results with a solution of San Francisco police district problem, as described in

John, Yohan, et al.
  "RoSSO: A High-Performance Python Package for Robotic Surveillance Strategy Optimization Using JAX."
arXiv preprint arXiv:2309.08742 (2023).
"""

import csv
import datetime
import time
import os

import networkx
from termcolor import colored
from fractions import Fraction

import stackelberg_games.patrolling as sgp

def run_series():
    """Set up and run computations."""
    
    tau = {0: 8, 1: 6, 2: 11, 3: 10, 4: 6, 5: 10, 6: 9, 7: 10, 8: 11, 9: 9, 10: 10, 11: 8}
    rollout_len = 10**3
    rollouts_num = 10**3
    reps = 1
    
    out_dir_path = 'output/sf_results/'
    os.makedirs(out_dir_path, exist_ok=True)

    with open(f'{out_dir_path}res_{round(datetime.datetime.timestamp(datetime.datetime.now()))}.csv', 'w', encoding='utf-8', newline='') as outputFile:
        out = csv.writer(outputFile, delimiter=';')

        def output_row(row):
            out.writerow(row)
            print(row)

        output_row(['san_francisco', reps])
        output_row(['graph_size', 'stat', 'value'])
        
        for n in range(3, 13):
            print(colored(f'Computing problem solution for n={n}.', 'green'))
            environment = sgp.john_et_al(n, self_loops = False)
            data = sgp.patrolling_problem(environment, 1, tau)
            solver = sgp.BisectSolver(10**-3, True)
            start_time = time.time()
            solution = solver.solve(data)
            end_time = time.time()
            output_row([n, 'lp_lower_bound', float(solution.lower_bound)])
            output_row([n, 'lp_upper_bound', float(solution.upper_bound)])
            output_row([n, 'lp_runtime', end_time - start_time])
            
            print(colored(f'Running Monte Carlo simulation for n={n}.', 'green'))
            mc_same = solution.monte_carlo_expected_reward(data.observation_length, rollout_len=rollout_len, rollouts_num=rollouts_num, disable_tqdm=True)
            output_row([n, 'mc_same', mc_same])
            mc_plusone = solution.monte_carlo_expected_reward(data.observation_length + 1, rollout_len=rollout_len, rollouts_num=rollouts_num, disable_tqdm = True)
            output_row([n, 'mc_plusone', mc_plusone])
            mc_minusone = solution.monte_carlo_expected_reward(data.observation_length - 1, rollout_len=rollout_len, rollouts_num=rollouts_num, disable_tqdm = True)
            output_row([n, 'mc_minusone', mc_minusone])

def main():
    run_series()

if __name__ == '__main__':
    main()
