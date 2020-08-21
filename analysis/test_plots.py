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

ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions'

EXCLUDE_EMISSION = ['CO', 'CO2', 'HC', 'NOx', 'PMx', 'angle', 'eclass', 'electricity', 'fuel', 'noise']

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

def print_arguments(args):

    print('Arguments (analysis/test_plots.py):')
    print('\tExperiment root folder: {0}\n'.format(args.experiment_root_folder))

def main(experiment_root_folder=None):

    print('\nRUNNING analysis/test_plots.py\n')

    if not experiment_root_folder:
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

    vehicles_appended = []
    throughputs = []

    mean_values_per_eval = []

    for csv_file in csv_files:

        print('Processing CSV file: {0}'.format(csv_file))
        
        # Load CSV data.
        df_csv = get_emissions(csv_file)

        df_per_vehicle = get_vehicles(df_csv)

        df_per_vehicle_mean = df_per_vehicle.mean()
        mean_values_per_eval.append({'speed': df_per_vehicle_mean['speed'],
                                     'delay': df_per_vehicle_mean['waiting'],
                                     'travel_time': df_per_vehicle_mean['total']})

        vehicles_appended.append(df_per_vehicle)

        df_throughput = get_throughput(df_csv)
        throughputs.append(df_throughput)


    df_vehicles_appended = pd.concat(vehicles_appended)
    df_throughputs_appended = pd.concat(throughputs)
    
    print(df_vehicles_appended.shape)
    print(df_throughputs_appended.shape)

    # Write mean values per eval into a csv file.
    df_mean_metrics_per_eval = pd.DataFrame(mean_values_per_eval)
    df_mean_metrics_per_eval.to_csv('{0}/mean_metrics_per_eval.csv'.format(output_folder_path),
                                    float_format='%.3f')

    """
        Waiting time & travel time.
    """
    # Describe waiting time.
    print('Waiting time:')
    df_stats = df_vehicles_appended['waiting'].describe()
    df_stats.to_csv('{0}/test_waiting_time_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_vehicles_appended['waiting'], hist=False, kde=True,
                kde_kws = {'linewidth': 3})

    plt.xlabel('Waiting time (s)')
    plt.ylabel('Density')
    plt.title('Waiting time')
    plt.savefig('{0}/test_waiting_time_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/test_waiting_time_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Describe travel time.
    print('Travel time:')
    df_stats = df_vehicles_appended['total'].describe()
    df_stats.to_csv('{0}/test_travel_time_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_vehicles_appended['total'], hist=False, kde=True,
                 kde_kws = {'linewidth': 3})

    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.title('Travel time')
    plt.savefig('{0}/test_travel_time_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/test_travel_time_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Describe vehicles' speed.
    print('Speed:')
    df_stats = df_vehicles_appended['speed'].describe()
    df_stats.to_csv('{0}/test_speed_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_vehicles_appended['speed'], hist=False, kde=True,
                 kde_kws = {'linewidth': 3})

    plt.xlabel('Average Speed (m/s)')
    plt.ylabel('Density')
    plt.title('Vehicles\' speed')
    plt.savefig('{0}/test_speeds_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/test_speeds_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Aggregate results per cycle.
    intervals = np.arange(0, df_vehicles_appended['finish'].max(), cycle_time)
    df_per_cycle = df_vehicles_appended.groupby(pd.cut(df_vehicles_appended["finish"], intervals)).mean()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = df_per_cycle['waiting'].values
    X = np.linspace(1, len(Y), len(Y))

    plt.plot(X,Y)
    plt.xlabel('Cycle')
    plt.ylabel('Average waiting time (s)')
    plt.title('Waiting time')
    plt.savefig('{0}/test_waiting_time.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/test_waiting_time.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = df_per_cycle['total'].values
    X = np.linspace(1, len(Y), len(Y))

    plt.plot(X,Y)
    plt.xlabel('Cycle')
    plt.ylabel('Average travel time (s)')
    plt.title('Travel time')
    plt.savefig('{0}/test_travel_time.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/test_travel_time.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Throughput.

        (throughput is calculated per cycle length)
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    intervals = np.arange(0, df_throughputs_appended['time'].max(), cycle_time)
    df = df_throughputs_appended.groupby(pd.cut(df_throughputs_appended["time"], intervals)).count()

    Y = df['time'].values
    X = np.linspace(1, len(Y), len(Y))

    plt.plot(X,Y)

    plt.xlabel('Cycle')
    plt.ylabel('#cars')
    plt.title('Throughput')

    plt.savefig('{0}/test_throughput.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/test_throughput.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    plt.close()

    # Get test eval json file from experiment root folder.
    json_file = Path(experiment_root_folder) / 'rollouts_test.json'
    print('JSON file path: {0}\n'.format(json_file))

    # Load JSON data.
    with open(json_file) as f:
        json_data = json.load(f)

    id = str(json_data['id'][0])

    """
        Rewards per intersection.
    """
    dfs_r = [pd.DataFrame(r) for r in json_data['rewards'][id]]

    df_concat = pd.concat(dfs_r)

    by_row_index = df_concat.groupby(df_concat.index)
    df_rewards = by_row_index.mean()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for col in df_rewards.columns:
        plt.plot(df_rewards[col].rolling(window=40).mean(), label=col)

    plt.xlabel('Cycle')
    plt.ylabel('Reward')
    plt.title('Rewards per intersection')
    plt.legend()

    plt.savefig('{0}/rewards_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/rewards_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    plt.close()

    """
        Total rewards.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    plt.plot(df_rewards.sum(axis=1))

    plt.xlabel('Cycle')
    plt.ylabel('Reward')
    plt.title('Cumulative reward')

    plt.savefig('{0}/rewards_total.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/rewards_total.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    
    plt.close()

    total_reward = df_rewards.to_numpy().sum()

    # Describe total system cumulative reward.
    pd.DataFrame([total_reward]).to_csv('{0}/cumulative_reward.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)

    """
        Actions per intersection.
    """
    dfs_a = [pd.DataFrame(r) for r in json_data['actions'][id]]

    df_concat = pd.concat(dfs_a)

    by_row_index = df_concat.groupby(df_concat.index)
    df_actions = by_row_index.mean()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for col in df_actions.columns:
        plt.plot(df_actions[col].rolling(window=40).mean(), label=col)

    plt.xlabel('Cycle')
    plt.ylabel('Action')
    plt.title('Actions per intersection')
    plt.legend()

    plt.savefig('{0}/actions_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/actions_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    plt.close()

    """
        Number of vehicles.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    dfs_veh = [pd.DataFrame(r) for r in json_data['vehicles'][id]]

    df_concat = pd.concat(dfs_veh)

    by_row_index = df_concat.groupby(df_concat.index)
    df_vehicles = by_row_index.mean()

    plt.plot(df_vehicles)

    plt.xlabel('Cycle')
    plt.ylabel('# Vehicles')
    plt.title('Number of vehicles')

    plt.savefig('{0}/vehicles.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/vehicles.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        Average vehicles' velocity.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    dfs_vels = [pd.DataFrame(r) for r in json_data['velocities'][id]]

    df_concat = pd.concat(dfs_vels)

    by_row_index = df_concat.groupby(df_concat.index)
    df_velocities = by_row_index.mean()

    plt.plot(df_velocities)

    plt.xlabel('Cycle')
    plt.ylabel('Average velocities')
    plt.title('Vehicles\' velocities')

    plt.savefig('{0}/velocities.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/velocities.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    plt.close()

if __name__ == "__main__":
    main()
