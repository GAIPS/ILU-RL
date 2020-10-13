import os
import json
import pandas as pd
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

from .utils import *

plt.style.use('ggplot')

ALTRL_HOME = os.environ['ALTRL_HOME']

EMISSION_PATH = \
    f'{ALTRL_HOME}/data/emissions'

FIGURE_X = 6.0
FIGURE_Y = 4.0

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script evaluates a traffic light system.
        """
    )
    parser.add_argument('experiment_root_folder', type=str, nargs='?',
                        help='Experiment root folder.')

    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_arguments(args):

    print('Arguments (analysis/test_plots.py):')
    print('\tExperiment root folder: {0}\n'.format(args.experiment_root_folder))


if __name__ == "__main__":

    print('\nRUNNING analysis/metrics_per_route.py\n')

    args = get_arguments()
    print_arguments(args)
    experiment_root_folder = args.experiment_root_folder

    # Prepare output folder.
    output_folder_path = os.path.join(experiment_root_folder, 'plots/test')
    print('Output folder: {0}'.format(output_folder_path))
    os.makedirs(output_folder_path, exist_ok=True)

    # Get cycle length from tls_config.json file.
    config_files = list(Path(experiment_root_folder).rglob('tls_config.json'))
    with config_files[0].open('r') as f:
        json_file = json.load(f)
    cycle_time = json_file['rl']['cycle_time']

    # Get all *.csv files from experiment root folder.
    csv_files = [str(p) for p in list(Path(experiment_root_folder).rglob('*-emission.csv'))]
    
    print('Number of csv files found: {0}'.format(len(csv_files)))

    for csv_file in csv_files:

        print('Processing CSV file: {0}'.format(csv_file))
        
        # Load CSV data.
        df_csv = get_emissions(csv_file)

        df_per_vehicle = get_vehicles(df_csv)

        # Randomly pick some routes.
        all_routes = df_per_vehicle['route'].unique()
        routes = np.random.choice(all_routes, 6)

        """
            PER-ROUTE METRICS PLOTS: ALL DAY .
        """
        # Speed per route - all day.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
             sns.distplot(df_per_vehicle[df_per_vehicle['route'] == route]['speed'], hist=False, kde=True,
                 kde_kws = {'linewidth': 3}, label=route)

        # All routes.
        sns.distplot(df_per_vehicle['speed'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average speed (m/s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average speed for different routes\n(Default)')
        plt.savefig('{0}/test_speeds_hist_per_route_default.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Travel time per route.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
            sns.distplot(df_per_vehicle[df_per_vehicle['route'] == route]['total'], hist=False, kde=True,
                 kde_kws = {'linewidth': 3}, label=route)

        sns.distplot(df_per_vehicle['total'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average travel time (s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average travel time for different routes\n(Default)')
        plt.savefig('{0}/test_travel_time_hist_per_route_default.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Waiting time per route.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
            sns.distplot(df_per_vehicle[df_per_vehicle['route'] == route]['waiting'], hist=False, kde=True,
                 kde_kws = {'linewidth': 3}, label=route)

        sns.distplot(df_per_vehicle['waiting'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average waiting time (s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average waiting time for different routes\n(Default)')
        plt.savefig('{0}/test_waiting_time_hist_per_route_default.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            PER-ROUTE METRICS PLOTS (FREE-FLOW ).
        """
        FREE_FLOW_TIME_CUT = 20000
        # Speed per route - free-flow.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
            sns.distplot(df_per_vehicle[(df_per_vehicle['route'] == route) & (df_per_vehicle['finish'] < FREE_FLOW_TIME_CUT)]['speed'], hist=False, kde=True,
                    kde_kws = {'linewidth': 3}, label=route)

        sns.distplot(df_per_vehicle[(df_per_vehicle['finish'] < FREE_FLOW_TIME_CUT)]['speed'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average speed (m/s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average speed for different routes\n(Free-flow)')
        plt.savefig('{0}/test_speeds_hist_per_route_free_flow.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Travel time per route - free-flow.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
            sns.distplot(df_per_vehicle[(df_per_vehicle['route'] == route) & (df_per_vehicle['finish'] < FREE_FLOW_TIME_CUT)]['total'], hist=False, kde=True,
                    kde_kws = {'linewidth': 3}, label=route)

        sns.distplot(df_per_vehicle[(df_per_vehicle['finish'] < FREE_FLOW_TIME_CUT)]['total'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average travel time (s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average travel time for different routes\n(Free-flow)')
        plt.savefig('{0}/test_travel_time_hist_per_route_free_flow.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Waiting time per route free-flow.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
            sns.distplot(df_per_vehicle[(df_per_vehicle['route'] == route) & (df_per_vehicle['finish'] < FREE_FLOW_TIME_CUT)]['waiting'], hist=False, kde=True,
                    kde_kws = {'linewidth': 3}, label=route)

        sns.distplot(df_per_vehicle[(df_per_vehicle['finish'] < FREE_FLOW_TIME_CUT)]['waiting'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average waiting time (s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average waiting time for different routes\n(Free-flow)')
        plt.savefig('{0}/test_waiting_time_hist_per_route_free_flow.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            PER-ROUTE METRICS PLOTS (CONGESTED).
        """
        CONGESTED_TIME_LOW_CUT = 25000
        CONGESTED_TIME_HIGH_CUT = 40000
        # Speed per route congested.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
            sns.distplot(df_per_vehicle[(df_per_vehicle['route'] == route) & (df_per_vehicle['finish'] < CONGESTED_TIME_HIGH_CUT) & (df_per_vehicle['finish'] > CONGESTED_TIME_LOW_CUT)]['speed'], hist=False, kde=True,
                    kde_kws = {'linewidth': 3}, label=route)


        sns.distplot(df_per_vehicle[(df_per_vehicle['finish'] < CONGESTED_TIME_HIGH_CUT) & (df_per_vehicle['finish'] > CONGESTED_TIME_LOW_CUT)]['speed'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average Speed (m/s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average speed for different routes\n(Congested)')
        plt.savefig('{0}/test_speeds_hist_per_route_congested.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Travel time per route congested.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
            
            sns.distplot(df_per_vehicle[(df_per_vehicle['route'] == route) & (df_per_vehicle['finish'] < CONGESTED_TIME_HIGH_CUT) & (df_per_vehicle['finish'] > CONGESTED_TIME_LOW_CUT)]['total'], hist=False, kde=True,
                    kde_kws = {'linewidth': 3}, label=route)

        sns.distplot(df_per_vehicle[(df_per_vehicle['finish'] < CONGESTED_TIME_HIGH_CUT) & (df_per_vehicle['finish'] > CONGESTED_TIME_LOW_CUT)]['total'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average travel time (s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average travel time for different routes\n(Congested)')
        plt.savefig('{0}/test_travel_time_hist_per_route_congested.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Waiting time per route congested.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for route in routes:
            sns.distplot(df_per_vehicle[(df_per_vehicle['route'] == route) & (df_per_vehicle['finish'] < CONGESTED_TIME_HIGH_CUT) & (df_per_vehicle['finish'] > CONGESTED_TIME_LOW_CUT)]['waiting'], hist=False, kde=True,
                    kde_kws = {'linewidth': 3}, label=route)

        sns.distplot(df_per_vehicle[(df_per_vehicle['finish'] < FREE_FLOW_TIME_CUT) & (df_per_vehicle['finish'] > CONGESTED_TIME_LOW_CUT)]['waiting'], hist=False, kde=True, color='black',
                 kde_kws = {'linewidth': 3, 'linestyle': '--'}, label='All routes')

        plt.legend()
        plt.xlabel('Average waiting time (s)')
        plt.ylabel('Density')
        plt.title('Vehicles\' average waiting time for different routes\n(Congested)')
        plt.savefig('{0}/test_waiting_time_hist_per_route_congested.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()
