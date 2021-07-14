"""This script helps setting sensible discretization bins

    This script performs the digitalization from continous spaces to discrete bins.
        * Splits results per network, tls and phase.
        * It searches for both train_logs and train_config for each experiment.
        * Aggregating MDPs when necessary for enrichment of datasets.
        * It creates a categories.json
"""
import re
from os import environ
from pathlib import Path
import json
from collections import defaultdict

import argparse
from scipy import stats
import configparser
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from ilurl.utils.digitize import digitize
from ilurl.utils.aux import snakefy
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# CONSTANTS
FIGURE_X = 6.0
FIGURE_Y = 4.0
ILURL_PATH = environ['ILURL_HOME']
NUM_PHASES = 2  #TODO: RELAX THIS ASSUMPTION

MATCHER = re.compile('\[(.*?)\]')

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script performs the digitalization from continous spaces to discrete bins.
                * Splits results per network, tls and phase.
                * It searches for both train_logs and train_config for each experiment.
                * Aggregating MDPs when necessary for enrichment of datasets.
                * It creates a categories.json
        """
    )
    parser.add_argument('experiment_collection_folder',
                        help='''Experiment collection folder.
                                Possibly for multi state-space definitions''')

    parser.add_argument('--percentiles', type=str2float, nargs="+", required=False,
                    help='Percentiles array which values range from 0.01 until 0.99', default=[0.10, 0.30, 0.50, 0.70, 0.90])

    parser.add_argument('--inverse', type=str2bool, nargs=1, required=False,
                    help='Prints percentiles compared to categories.json', default=False)

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/digitize):')
    print(f'\tExperiment collection folder: {args.experiment_collection_folder}\n')
    print(f'\tPercentiles: {args.percentiles}\n')

def str2float(v):
    return eval(v)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    print('\nRUNNING analysis/digitize.py\n')

    args = get_arguments()
    print_arguments(args)

    if Path(args.experiment_collection_folder).suffix == '.gz':
        raise ValueError('Please uncompress folder first.')
    origin_path = Path(args.experiment_collection_folder)

    train_log_paths = list(p for p in origin_path.rglob('*train_log.json'))

    percentiles = args.percentiles
    inverse = args.inverse

    # I) Data extraction loop
    # Loads experiments into memory
    print('################################################################################')
    print(f'Training logs found {len(train_log_paths)}.')
    print('Loading...')

    target_path = origin_path / 'category_plots'
    target_path.mkdir(mode=0o777, exist_ok=True)
    if inverse:
        # III) Data benchmark loop
        # compares category json with experiments' percentiles.
        data_path = Path(ILUTRL_PATH) / 'data' / 'networks'
        for network in networks:

            categories_path = data_path / network / 'categories.json'
            data2 = {k: v for k, v in data1.items() if network == k[0]}
            tlids = sorted({kk for dat in data2.values() for kk in dat})

            data3 = {} # categories json
            data4 = defaultdict(dict) # quantile CSV
            with categories_path.open(mode='r') as f:
                categories = json.load(f)

            for tid in tlids:
                for key, values in data2.items():
                    _, feature = key

                    q1, q2 = categories[tid][feature].values()
                    d1, d2 = zip(*values[tid]) # should have two phases


                    # (i) compute the scores from the quantiles
                    # (ii) compute the scores from the quantiles
                    used_p1 = np.quantile(d1, percentiles).tolist()
                    scores_p1 = [stats.percentileofscore(d1, x) / 100  for x in q1]
                    sampled_p1 = np.quantile(d1, scores_p1)

                    used_p2 = np.quantile(d2, percentiles).tolist()
                    scores_p2 = [stats.percentileofscore(d2, x) / 100 for x in q2]
                    sampled_p2 = np.quantile(d2, scores_p2)

                    # assumptions there are always two phases
                    labels = [f'{feature}_{n}' for n in range(2)]
                    fig, ax = plt.subplots()
                    fig.set_size_inches(FIGURE_X, FIGURE_Y)

                    sns.distplot(d1, hist=True, kde=True,
                                 label=labels[0], kde_kws = {'linewidth': 3})
                    ax.vlines(used_p1, 0, 1,
                              transform=ax.get_xaxis_transform(),
                              colors='tab:purple', label=f'Used bins')
                    ax.vlines(sampled_p1, 0, 1,
                              transform=ax.get_xaxis_transform(),
                              colors='tab:cyan', label=f'Sample bins')

                    plt.xlabel(f'{snakefy(feature)}')
                    plt.ylabel('Density')
                    plt.title(f'{snakefy(network)}: Intersection {tid}, {snakefy(feature)} feature')

                    plt.legend((feature, 'Used bins', 'Sampled bins'))
                    plt.savefig((target_path / f'{tid}-hist-{feature}-phase1.png').as_posix(),
                                bbox_inches='tight', pad_inches=0)
                    plt.close()

                    fig, ax = plt.subplots()
                    fig.set_size_inches(FIGURE_X, FIGURE_Y)
                    sns.distplot(d2, hist=True, kde=True,
                                 label=labels[1], kde_kws = {'linewidth': 3})
                    ax.vlines(used_p2, 0, 1,
                              transform=ax.get_xaxis_transform(),
                              colors='tab:purple', label=f'Used bins')
                    ax.vlines(sampled_p2, 0, 1,
                              transform=ax.get_xaxis_transform(),
                              colors='tab:cyan', label=f'Sample bins')

                    plt.xlabel(f'{snakefy(feature)}')
                    plt.ylabel('Density')
                    plt.title(f'{snakefy(network)}: Intersection {tid}, {snakefy(feature)} feature')

                    plt.legend((feature, 'Used bins', 'Sampled bins'))
                    plt.savefig((target_path / f'{tid}-hist-{feature}-phase2.png').as_posix(),
                                bbox_inches='tight', pad_inches=0)
                    plt.close()
    else:
        data2, networks = digitize(train_log_paths)
        data4 = defaultdict(dict)
        assert len(networks) == 1, 'number of networks must equal to 1.'
        for network in networks:
            for tid, features in data2.items():
                for feature, scores in features.items():
                    assert len(scores) == 2, 'Number of phases must equal to 2.'

                    labels = [f'{feature}_{n}' for n in range(2)]
                    fig, ax = plt.subplots()
                    fig.set_size_inches(FIGURE_X, FIGURE_Y)
                    for p, s1, s2 in zip(percentiles, *scores.values()):
                        data4[(tid, feature, 0)][p] = s1
                        data4[(tid, feature, 1)][p] = s2

                        sns.distplot(s1, hist=False, kde=True,
                                     label=labels[0], kde_kws = {'linewidth': 3})
                        sns.distplot(s2, hist=False, kde=True,
                                     label=labels[1], kde_kws = {'linewidth': 3})

                        plt.xlabel(f'{snakefy(feature)}')
                        plt.ylabel('Density')
                        plt.title(f'{snakefy(network)}: {tid}, {snakefy(feature)} feature')
                        plt.savefig((target_path / f'{tid}-{feature}.png').as_posix(),
                                    bbox_inches='tight', pad_inches=0)
                        plt.close()

        # Save json categories
        with (target_path / 'categories.json').open('w') as f:
            json.dump(data2, f)

        # Save quantile tabular data
        df = pd.DataFrame.from_dict(data4).transpose()
        df.to_csv((target_path / 'quantile.csv').as_posix(), sep=',', encoding='utf-8')


if __name__ == '__main__':
    main()


