import os
import tarfile
import pandas as pd
import argparse
from pathlib import Path
import configparser
import tempfile
import shutil

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
            This script creates a kde dist plot that allows comparisons between \
                the mean distributions of different experiments.
        """
    )
    parser.add_argument('--experiments_paths', nargs="+", help='List of paths to experiments.', required=True)
    parser.add_argument('--labels', nargs="+", help='List of experiments\' labels.', required=False)

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/means_dist.py):')
    print('\tExperiments: {0}\n'.format(args.experiments_paths))
    print('\tExperiments labels: {0}\n'.format(args.labels))


def main():

    print('\nRUNNING analysis/means_dist.py\n')

    args = get_arguments()
    print_arguments(args)

    # Prepare output folder.
    os.makedirs('analysis/plots/means_dist/', exist_ok=True)

    # Open dataframes.
    dfs = []
    for exp_path in args.experiments_paths:

        if Path(exp_path).suffix == '.gz':
            # Compressed file (.tar.gz).

            tar = tarfile.open(exp_path)
            all_names = tar.getnames()

            # Get one of the config files.
            config_files = [x for x in all_names if Path(x).name == 'train.config']
            config_p = config_files[0]

            # Create temporary directory.
            dirpath = tempfile.mkdtemp()

            # Extract config file to temporary directory.
            tar.extract(config_p, dirpath)

            train_config = configparser.ConfigParser()
            train_config.read(dirpath + '/' + config_p)

            # Print config file.
            tls_type = train_config['train_args']['tls_type']

            # Clean temporary directory.
            shutil.rmtree(dirpath)

            exp_name = os.path.basename(exp_path)
            exp_name = exp_name.split('.')[0] + '.' + exp_name.split('.')[1]
            df = pd.read_csv(tarfile.open(exp_path).extractfile(
                                    f'{exp_name}/plots/test/{exp_name}_metrics.csv'))

            if tls_type == 'rl':
                df = df.groupby(['train_run']).mean()

            dfs.append(df)

        else:
            # Uncompressed file (experiment_folder).
            raise ValueError('Not implemented for uncompressed folders. Please point to a file with .tar.gz extension.')

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

    for df, lbl in zip(dfs, lbls):
        sns.distplot(df['waiting_time'], hist=False, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label=lbl)

    plt.legend()
    plt.xlabel('Waiting time (s)')
    plt.ylabel('Density')
    plt.savefig('analysis/plots/means_dist/waiting_time.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/means_dist/waiting_time.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Travel time.
    """
    fig = plt.figure()

    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for df, lbl in zip(dfs, lbls):
        sns.distplot(df['travel_time'], hist=False, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label=lbl)

    plt.legend()
    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.savefig('analysis/plots/means_dist/travel_time.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/means_dist/travel_time.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Speed.
    """
    fig = plt.figure()

    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for df, lbl in zip(dfs, lbls):
        sns.distplot(df['speed'], hist=False, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label=lbl)

    plt.legend()
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Density')
    plt.savefig('analysis/plots/means_dist/speed.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/means_dist/speed.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    main()