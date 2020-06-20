"""
    Python script to produce the following (global metrics) train plots:
        - reward per cycle (with mean, std and smoothed curve)
        - number of vehicles per cycle (with mean, std and smoothed curve)
        - vehicles' velocity per cycle (with mean, std and smoothed curve)

    Given the path to the experiment root folder (-p flag), the script
    searches recursively for all train_log.json files and produces the
    previous plots by averaging over all json files.

    The output plots will go into a folder named 'plots', created inside
    the given experiment root folder.
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path

import pandas as pd

import statsmodels.api as sm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


FIGURE_X = 6.0
FIGURE_Y = 4.0

STD_CURVE_COLOR = (0.88,0.70,0.678)
MEAN_CURVE_COLOR = (0.89,0.282,0.192)
SMOOTHING_CURVE_COLOR = (0.33,0.33,0.33)


def get_arguments():

    parser = argparse.ArgumentParser(
        description="""
            Python script to produce the following (global metrics) train plots:
                - reward per cycle (with mean, std and smoothed curve)
                - number of vehicles per cycle (with mean, std and smoothed curve)
                - vehicles' velocity per cycle (with mean, std and smoothed curve)

            Given the path to the experiment root folder (-p flag), the script
            searches recursively for all train_log.json files and produces the
            previous plots by averaging over all json files.

            The output plots will go into a folder named 'plots', created inside
            the given experiment root folder.
        """
    )

    parser.add_argument('--path', '-p', type=str, nargs='?',
                dest='experiment_root_folder', required=True,
                help='Path to the experiment root folder')

    return parser.parse_args()


def main(experiment_root_folder=None):

    print('\nRUNNING analysis/train_plots.py\n')

    if not experiment_root_folder:
        args = get_arguments()
        experiment_root_folder = args.experiment_root_folder

    print('Input files:')
    # Get all train_log.json files from experiment root folder.
    train_files = []
    for path in Path(experiment_root_folder).rglob('train_log.json'):
        train_files.append(str(path))
        print('{0}'.format(str(path)))

    # Prepare output folder.
    output_folder_path = os.path.join(experiment_root_folder, 'plots/train')
    print('\nOutput folder:\n{0}\n'.format(output_folder_path))
    os.makedirs(output_folder_path, exist_ok=True)

    actions = []
    rewards = []
    vehicles = []
    velocities = []

    # Concatenate data for all runs.
    for run_name in train_files:

        # Load JSON data.
        with open(run_name) as f:
            json_data = json.load(f)

        # Rewards per time-step.
        r = json_data['rewards']
        r = pd.DataFrame(r)
        rewards.append(np.sum(r.values, axis=1))

        # Number of vehicles per time-step.
        vehicles.append(json_data['vehicles'])

        # Vehicles' velocity per time-step.
        velocities.append(json_data['velocities'])

        # Agent's actions.
        actions.append(json_data['actions'])

    """
        Rewards per cycle.
        (GLOBAL: sum of the reward for all intersections).
    """
    rewards = np.array(rewards)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(rewards, axis=0)
    Y_std = np.std(rewards, axis=0)
    X = np.linspace(1, rewards.shape[1], rewards.shape[1])

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if rewards.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Train cycle')
    plt.ylabel('Reward')
    plt.title('Train rewards ({0} runs)'.format(len(train_files)))
    plt.legend(loc=4)

    file_name = '{0}/train_rewards.pdf'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = '{0}/train_rewards.png'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """ 
        Number of vehicles per cycle.
        (GLOBAL: For all vehicles in simulation)
    """
    vehicles = np.array(vehicles)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(vehicles, axis=0)
    Y_std = np.std(vehicles, axis=0)
    X = np.linspace(1, vehicles.shape[1], vehicles.shape[1])

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if vehicles.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Train cycle')
    plt.ylabel('Train: #Vehicles')
    plt.title('Number of vehicles ({0} runs)'.format(len(train_files)))
    plt.legend(loc=4)

    file_name = '{0}/train_vehicles.pdf'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = '{0}/train_vehicles.png'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """ 
        Vehicles' velocity per cycle.
        (GLOBAL: For all vehicles in simulation)
    """
    velocities = np.array(velocities)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(velocities, axis=0)
    Y_std = np.std(velocities, axis=0)
    X = np.linspace(1, velocities.shape[1], velocities.shape[1])

    # Replace NaNs.
    Y_lowess = np.where(np.isnan(Y), 0, Y)

    lowess = sm.nonparametric.lowess(Y_lowess, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if velocities.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Train cycle')
    plt.ylabel('Velocity')
    plt.title('Train: Velocity of the vehicles ({0} runs)'.format(len(train_files)))
    plt.legend(loc=4)

    file_name = '{0}/train_velocities.pdf'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = '{0}/train_velocities.png'.format(output_folder_path)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """ 
        Actions per intersection.
    """
    dfs_a = [pd.DataFrame(a) for a in actions]

    df_concat = pd.concat(dfs_a)

    by_row_index = df_concat.groupby(df_concat.index)
    df_actions = by_row_index.mean()

    print(df_actions)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    window_size = min(len(df_actions)-1, 40)

    print(window_size)

    for col in df_actions.columns:
        plt.plot(df_actions[col].rolling(window=window_size).mean(), label=col)

    plt.xlabel('Cycle')
    plt.ylabel('Action')
    plt.title('Actions per intersection')
    plt.legend()

    plt.savefig('{0}/actions_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/actions_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    plt.close()

if __name__ == '__main__':
    main()