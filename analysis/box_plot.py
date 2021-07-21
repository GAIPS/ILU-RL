import os
import tarfile
import pandas as pd
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script creates a box plot that allows comparisons between different experiments.
        """
    )
    parser.add_argument('--experiments_paths', nargs="+", help='List of paths to experiments.', required=True)
    parser.add_argument('--labels', nargs="+", help='List of experiments\' labels.', required=False)

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/box_plot.py):')
    print('\tExperiments: {0}\n'.format(args.experiments_paths))
    print('\tExperiments labels: {0}\n'.format(args.labels))

def main():

    print('\nRUNNING analysis/box_plot.py\n')

    args = get_arguments()
    print_arguments(args)

    # Prepare output folder.
    os.makedirs('analysis/plots/box_plots/', exist_ok=True)

    # Open dataframes.
    dfs_waiting_time = []
    dfs_travel_time = []
    dfs_speed = []
    for exp_path in args.experiments_paths:

        if Path(exp_path).suffix == '.gz':
            # Compressed file (.tar.gz).

            exp_name = Path(exp_path).name.split('.')[0] + \
                        '.' + Path(exp_path).name.split('.')[1]

            tar = tarfile.open(exp_path)
            tar_file_wt = tar.extractfile("{0}/plots/test/waiting_time_stats.csv".format(exp_name))
            tar_file_s = tar.extractfile("{0}/plots/test/speed_stats.csv".format(exp_name))
            tar_file_tt = tar.extractfile("{0}/plots/test/travel_time_stats.csv".format(exp_name))

            dfs_waiting_time.append(pd.read_csv(tar_file_wt, header=[0, 1], index_col=0))
            dfs_travel_time.append(pd.read_csv(tar_file_tt, header=[0, 1], index_col=0))
            dfs_speed.append(pd.read_csv(tar_file_s, header=[0, 1], index_col=0))

        else:
            # Uncompressed file (experiment_folder).
            exp_name = Path(exp_path).name
            dfs_waiting_time.append(pd.read_csv('{0}/plots/test/waiting_time_stats.csv'.format(
                                            exp_path), header=[0, 1], index_col=0))
            dfs_travel_time.append(pd.read_csv('{0}/plots/test/travel_time_stats.csv'.format(
                                            exp_path), header=[0, 1], index_col=0))
            dfs_speed.append(pd.read_csv('{0}/plots/test/speed_stats.csv'.format(
                                            exp_path), header=[0, 1], index_col=0))

    if args.labels:
        lbls = args.labels # Custom labels.
    else:
        lbls = [Path(exp_path).name for exp_path in args.experiments_paths] # Default labels.

    """
        Waiting time.
    """
    fig = plt.figure()

    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    boxes = []
    for (df, lbl) in zip(dfs_waiting_time, lbls):
        boxes.append({
            'label' : lbl,
            'whislo': df.iloc[-5,0],    # Bottom whisker position
            'q1'    : df.iloc[-4,0],    # First quartile (25th percentile)
            'med'   : df.iloc[-3,0],    # Median         (50th percentile)
            'q3'    : df.iloc[-2,0],    # Third quartile (75th percentile)
            'whishi': df.iloc[-1,0],    # Top whisker position
            'fliers': []                # Outliers
        })

    ax.bxp(boxes)

    plt.xticks(rotation=45)

    plt.ylabel('Waiting time (s)')
    
    plt.savefig('analysis/plots/box_plots/waiting_time.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/box_plots/waiting_time.png', bbox_inches='tight', pad_inches=0)

    plt.close()

    """
        Travel time.
    """
    fig = plt.figure()

    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    boxes = []
    for (df, lbl) in zip(dfs_travel_time, lbls):
        boxes.append({
            'label' : lbl,
            'whislo': df.iloc[-5,0],    # Bottom whisker position
            'q1'    : df.iloc[-4,0],    # First quartile (25th percentile)
            'med'   : df.iloc[-3,0],    # Median         (50th percentile)
            'q3'    : df.iloc[-2,0],    # Third quartile (75th percentile)
            'whishi': df.iloc[-1,0],    # Top whisker position
            'fliers': []                # Outliers
        })

    ax.bxp(boxes, showfliers=False)

    plt.ylabel('Travel time (s)')

    plt.xticks(rotation=45)
    
    plt.savefig('analysis/plots/box_plots/travel_time.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/box_plots/travel_time.png', bbox_inches='tight', pad_inches=0)

    plt.close()

    """
        Speed.
    """
    fig = plt.figure()

    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    boxes = []
    for (df, lbl) in zip(dfs_speed, lbls):
        boxes.append({
            'label' : lbl,
            'whislo': df.iloc[-5,0],    # Bottom whisker position
            'q1'    : df.iloc[-4,0],    # First quartile (25th percentile)
            'med'   : df.iloc[-3,0],    # Median         (50th percentile)
            'q3'    : df.iloc[-2,0],    # Third quartile (75th percentile)
            'whishi': df.iloc[-1,0],    # Top whisker position
            'fliers': []                # Outliers
        })

    ax.bxp(boxes, showfliers=False)

    plt.ylabel('Speed (m/s)')

    plt.xticks(rotation=45)
    
    plt.savefig('analysis/plots/box_plots/speed.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/box_plots/speed.png', bbox_inches='tight', pad_inches=0)

    plt.close()

if __name__ == "__main__":
    main()