import os
import tarfile
import json
import pandas as pd
import numpy as np
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

from scipy import stats
import statsmodels
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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

    print('Arguments (analysis/stat_test.py):')
    print('\tExperiments: {0}\n'.format(args.experiments_paths))
    print('\tExperiments labels: {0}\n'.format(args.labels))


if __name__ == '__main__':

    print('\nRUNNING analysis/stat_test.py\n')

    args = get_arguments()
    print_arguments(args)

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


    # Shapiro tests.
    print('Shapiro tests:')
    for (df, lbl) in zip(dfs,lbls):
        shapiro_test = stats.shapiro(df['travel_time'])
        print(f'\t{lbl}: {shapiro_test}')

    args = [df['travel_time'] for df in dfs]
    print(f'\nLevene\'s test: {stats.levene(*args)}')

    print(f'\nANOVA test: {stats.f_oneway(*args)}')

    data = []
    groups = []
    for (df, lbl) in zip(dfs, lbls):
        data.extend(df['travel_time'].tolist())
        groups.extend([lbl for _ in range(len(df['travel_time'].tolist()))])

    print('\nTukeyHSD:', pairwise_tukeyhsd(data, groups))

    # Non-parametric test.
    print('\nKruskal (non-parametric) test:', stats.kruskal(*args))

    # Post-hoc non-parametric comparisons.
    data = [df['travel_time'].tolist() for df in dfs]
    print(sp.posthoc_conover(data, p_adjust = 'holm'))
