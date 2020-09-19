import os
import tarfile
import pandas as pd
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

from analysis.utils import str2bool, get_emissions, get_vehicles, get_throughput

plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script creates evaluation plots that allow comparisons between different experiments.
        """
    )
    parser.add_argument('--experiments_paths', nargs="+", help='List of paths to experiments.', required=True)
    parser.add_argument('--labels', nargs="+", help='List of experiments\' labels.', required=False)

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/compare.py):')
    print('\tExperiments: {0}\n'.format(args.experiments_paths))
    print('\tExperiments labels: {0}\n'.format(args.labels))

def main():

    print('\nRUNNING analysis/compare.py\n')

    args = get_arguments()
    print_arguments(args)

    # Prepare output folder.
    os.makedirs('analysis/plots/compare/', exist_ok=True)

    # Open dataframes.
    dfs = {}
    for exp_path in args.experiments_paths:

        if Path(exp_path).suffix == '.gz':
            # Compressed file (.tar.gz).

            exp_name = Path(exp_path).name.split('.')[0] + \
                        '.' + Path(exp_path).name.split('.')[1]

            tar = tarfile.open(exp_path)
            tar_file = tar.extractfile("{0}/plots/test/processed_data.csv".format(exp_name))
            
            dfs[exp_name] = pd.read_csv(tar_file, header=[0, 1], index_col=0)

        else:
            # Uncompressed file (experiment_folder).
            exp_name = Path(exp_path).name
            dfs[exp_name] = pd.read_csv('{0}/plots/test/processed_data.csv'.format(
                                            exp_path), header=[0, 1], index_col=0)

    if args.labels:
        lbls = args.labels # Custom labels.
    else:
        lbls = dfs.keys()  # Default labels.

    """
        waiting_time_hist_kde
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, dfs.values()):
        plt.plot(df['waiting_time_hist_kde', 'x'],
                df['waiting_time_hist_kde', 'y'],
                label=l, linewidth=3)

    plt.xlabel('Waiting time (s)')
    plt.legend()
    plt.ylabel('Density')
    plt.title('Waiting time')
    
    plt.savefig('analysis/plots/compare/waiting_time_hist.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/compare/waiting_time_hist.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        travel_time_hist_kde
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, dfs.values()):
        plt.plot(df['travel_time_hist_kde', 'x'],
                 df['travel_time_hist_kde', 'y'],
                 label=l, linewidth=3)

    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Travel time')
    
    plt.savefig('analysis/plots/compare/travel_time_hist.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/compare/travel_time_hist.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        speed_hist_kde
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, dfs.values()):
        plt.plot(df['speed_hist_kde', 'x'],
                 df['speed_hist_kde', 'y'],
                 label=l, linewidth=3)

    plt.xlabel('Average speed (m/s)')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Vehicles\' speed')
    
    plt.savefig('analysis/plots/compare/speeds_hist.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/compare/speeds_hist.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        waiting_time_per_cycle
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, dfs.values()):
        plt.plot(df['waiting_time_per_cycle', 'x'],
                 df['waiting_time_per_cycle', 'y'],
                 label=l)

    plt.xlabel('Cycle')
    plt.ylabel('Average waiting time (s)')
    plt.legend()
    plt.title('Waiting time')
    
    plt.savefig('analysis/plots/compare/waiting_time.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/compare/waiting_time.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        travel_time_per_cycle
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, dfs.values()):
        plt.plot(df['travel_time_per_cycle', 'x'],
                 df['travel_time_per_cycle', 'y'],
                 label=l)

    plt.xlabel('Cycle')
    plt.ylabel('Average travel time (s)')
    plt.legend()
    plt.title('Travel time')

    plt.savefig('analysis/plots/compare/travel_time.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/compare/travel_time.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        throughput_per_cycle
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, dfs.values()):
        plt.plot(df['throughput_per_cycle', 'x'],
                 df['throughput_per_cycle', 'y'],
                 label=l)

    plt.xlabel('Cycle')
    plt.ylabel('#cars')
    plt.legend()
    plt.title('Throughput')

    plt.savefig('analysis/plots/compare/throughput.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/compare/throughput.png', bbox_inches='tight', pad_inches=0)

    plt.close()

    """
        vehicles_per_cycle
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, dfs.values()):
        plt.plot(df['vehicles_per_cycle', 'x'],
                 df['vehicles_per_cycle', 'y'],
                 label=l)

    plt.xlabel('Cycle')
    plt.ylabel('# Vehicles')
    plt.title('Number of vehicles')
    plt.legend()

    plt.savefig('analysis/plots/compare/vehicles.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/compare/vehicles.png', bbox_inches='tight', pad_inches=0)

    plt.close()

    """
        velocities_per_cycle
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for l, df in zip(lbls, dfs.values()):
        plt.plot(df['velocities_per_cycle', 'x'],
                 df['velocities_per_cycle', 'y'],
                 label=l)

    plt.xlabel('Cycle')
    plt.ylabel('Average velocities')
    plt.title('Vehicles\' velocities')
    plt.legend()

    plt.savefig('analysis/plots/compare/velocities.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/compare/velocities.png', bbox_inches='tight', pad_inches=0)

    plt.close()


if __name__ == "__main__":
    main()