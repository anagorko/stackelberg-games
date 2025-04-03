import os
import pickle

import stackelberg_games.patrolling as sgp


def solve_and_save_defense_plan(elu: int, units: int, obs: int, at: int):
    out_dir_path = sgp.directories.strategy_dir
    os.makedirs(out_dir_path, exist_ok=True)

    output_name = f"{out_dir_path}gdynia_{elu}_{units}_{obs}_{at}.pickle"

    if os.path.exists(output_name):
        print(f"{output_name} already exists, skipping.")
        return

    gp = sgp.gdynia_problem(edge_len_unit=elu, number_of_units=units, observation_length=obs, att_time=at)
    solver = sgp.BisectSolver()
    solution = solver.solve(gp)
    with open(output_name, "wb") as output_file:
        pickle.dump(solution, output_file)
        print(f"Wrote strategy to {output_name}.")


def main():
    solve_and_save_defense_plan(1000, 1, 1, 3)  # Utility 0.25 >= 1/4
    solve_and_save_defense_plan(1000, 1, 2, 3)  # Utility 0.19 >= 3/16
    solve_and_save_defense_plan(1000, 1, 3, 3)  # Utility
    solve_and_save_defense_plan(1000, 1, 4, 3)  # Utility
    solve_and_save_defense_plan(1000, 1, 5, 3)  # Utility
    solve_and_save_defense_plan(1000, 2, 1, 3)  # Utility 0.95 >= 61/64


if __name__ == '__main__':
    main()
