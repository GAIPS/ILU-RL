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
import configparser
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from ilurl.utils.aux import snakefy
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# CONSTANTS
FIGURE_X = 6.0
FIGURE_Y = 4.0
ILU_PATH = environ['ILURL_HOME']
NUM_PHASES = 2  #TODO: RELAX THIS ASSUMPTION

MATCHER = re.compile('\[(.*?)\]')

def rmlag(x):
    if 'lag[' in x:
        return MATCHER.search(x).groups()[0]
    return x

def fn(cycles, ind):
    return [
                {
                    tid: [phase[ind] for phase in phases]
                    for tid, phases in cycle.items()
                }
                for cycle in cycles
            ]

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
                    help='Percentiles array which values range from 0.01 until 0.99', default=[0.15, 0.5, 0.85])

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/digitize):')
    print(f'\tExperiment collection folder: {args.experiment_collection_folder}\n')
    print(f'\tPercentiles: {args.percentiles}\n')

def str2float(v):
    return eval(v)

def main():

    print('\nRUNNING analysis/digitize.py\n')

    args = get_arguments()
    print_arguments(args)

    if Path(args.experiment_collection_folder).suffix == '.gz':
        raise ValueError('Please uncompress folder first.')
    origin_path = Path(args.experiment_collection_folder)

    train_log_paths = list(p for p in origin_path.rglob('*train_log.json'))

    percentiles = args.percentiles
    # raise ValueError('num_samples argument should be <= than the number of training runs.')

    # I) Data extraction loop
    # Loads experiments into memory
    print('################################################################################')
    print(f'Training logs found {len(train_log_paths)}.')
    print('Loading...')
    data = defaultdict(list)
    for train_log_path in train_log_paths:

        # Read config and common 
        with train_log_path.open('r') as f:
            train_data = json.load(f)
        train_data = train_data['observation_spaces']

        # Read config and common 
        config_path = train_log_path.parent.parent / 'config' / 'train.config'

        train_config = configparser.ConfigParser()
        train_config.read(config_path)

        # TODO: go to config and network
        network = train_config.get('train_args', 'network')
        features = eval(train_config.get('mdp_args', 'features'))

        # Remove lag from features
        features = tuple(rmlag(f) for f in features)
        key = (network, features)
        data[key] += train_data

    # II) Data aggregation loop
    # paginates all tuples consolidating same features.
    # i.e 'count' from mdp speed_delta and speed_score are aggregated
    # i.e 'lag[.]' states are excluded
    data1 = {}
    print('Merging experiments...')
    for k, v in data.items():
        network, features = k
        for feature in features:
            if (network, feature) not in data1:
                data1[(network, feature)] = defaultdict(list)
                sel = {k1: fn(v1, k1[1].index(feature))
                       for k1, v1 in data.items() if k1[0] == network and feature in k1[1]}
                num_points =[]
                for cycles in sel.values():
                    for cycle in cycles:
                        for tl, phases in cycle.items():
                            data1[(network, feature)][tl].append(phases)
                            num_points.append(len(phases))
                print(f'{network}\tnum. nodes: {len(data1[(network, feature)])}\tfeature: {feature}\tnum_phases: {len(num_points)}\tnum. data points:{sum(num_points)}.')

    tlids = sorted({k for dat in data1.values() for k in dat})
    networks = sorted({k[0] for dat in data1})

    # III) Data partitioning loop
    # performs quantization of histograms.
    target_path = origin_path / 'category_plots'
    target_path.mkdir(mode=0o777, exist_ok=True)
    for network in networks:
        network_path = target_path / network
        network_path.mkdir(mode=0o777, exist_ok=True)

        data2 = {k: v for k, v in data1.items() if network == k[0]}
        tlids = sorted({kk for dat in data2.values() for kk in dat})

        dest_path = network_path / 'categories.json'
        data3 = {} # categories json
        data4 = defaultdict(dict) # quantile CSV
        with dest_path.open(mode='w') as f:

            for tid in tlids:
                data3[tid] = {}
                for key, values in data2.items():
                    _, feature = key
                    d1, d2 = zip(*values[tid]) # should have two phases
                    p1 = np.quantile(d1, percentiles).tolist()
                    p2 = np.quantile(d2, percentiles).tolist()
                    data3[tid][feature] = {'0': p1, '1': p2}
                    for p, pp1, pp2 in zip(percentiles, p1, p2):
                        data4[(tid, feature, 0)][p] = pp1
                        data4[(tid, feature, 1)][p] = pp2

                    # assumptions there are always two phases
                    labels = [f'{feature}_{n}' for n in range(2)]
                    fig, ax = plt.subplots()
                    fig.set_size_inches(FIGURE_X, FIGURE_Y)

                    sns.distplot(d1, hist=False, kde=True,
                                 label=labels[0], kde_kws = {'linewidth': 3})
                    sns.distplot(d2, hist=False, kde=True,
                                 label=labels[1], kde_kws = {'linewidth': 3})

                    plt.xlabel(f'{snakefy(feature)}')
                    plt.ylabel('Density')
                    plt.title(f'{snakefy(network)}: Intersection {tid}, {snakefy(feature)} feature')
                    plt.savefig((target_path / f'{tid}-{feature}.png').as_posix(),
                                bbox_inches='tight', pad_inches=0)
                    plt.close()

                # Save json categories
                with (target_path / 'categories.json').open('w') as f:
                    json.dump(data3, f)

                # Save quantile tabular data
                df = pd.DataFrame.from_dict(data4).transpose()
                df.to_csv((target_path / 'quantile.csv').as_posix(), sep=',', encoding='utf-8')


if __name__ == '__main__':
    main()


