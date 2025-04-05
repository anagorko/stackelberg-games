"""
Animation for IDA.
"""

from __future__ import annotations

import argparse
from termcolor import colored

import stackelberg_games.patrolling as sgp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', default=sgp.animate_defaults['defense_plan'], nargs='?',
                        help='Name of pickled DefensePlan.')
    args = parser.parse_args()

    out_dir_path = sgp.directories.strategy_dir

    animation_group = sgp.AnimationGroup(
        map_tags=['osm', 'google_terrain', 'stadia_smooth'],
        defense_plan_filenames=[
            f'{out_dir_path}gdynia_1000_1_1_3.pickle',
            #f'{out_dir_path}gdynia_1000_1_2_3.pickle',
            #f'{out_dir_path}gdynia_1000_1_3_3.pickle',
            f'{out_dir_path}gdynia_1000_1_4_3.pickle',
            #f'{out_dir_path}gdynia_1000_1_5_3.pickle',
            f'{out_dir_path}gdynia_1000_2_1_3.pickle'
        ],
        speeds=[2, 5],
        steps=[30, 12]
    )

    for animation in animation_group.animations:
        print('Processing', colored(animation.output_filename, 'green'))
        animation.render()


if __name__ == '__main__':
    main()
