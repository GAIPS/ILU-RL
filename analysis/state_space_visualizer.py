"""This script helps setting sensible discretization bins

    * Run an experiment such as DQN.
    * Change path to train_log_folder/
      (i.e data/emissions/20200916183156.533079/grid_6_20200916-1831561600277516.627527/logs/train_log.json)

    * Will create experiment_folder/log_plots.
    * Will save plots for histograms.
    * Will save quantiles for distributions.

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
# TODO: make this an input argument
# LOG_PATH = 'data/emissions/20200916183156.533079/grid_6_20200916-1831561600277516.627527/logs/train_log.json'
# LOG_PATH = 'data/emissions/20200828015100.697569/grid_20200828-0151021598575862.5034564/logs/train_log.json'
# LOG_PATH = "data/emissions/20200916161353.347374/grid_6_20200916-1613531600269233.4435577/logs/train_log.json"

# LOG_PATH = 'data/experiments/intbins/wtime/intersection_20201009-2004551602270295.8987248/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/ss/intersection_20201009-2344491602283489.5916183/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/sc/intersection_20201009-1125051602239105.9357407/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/ddel/intersection_20201012-1944331602528273.319772/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/del/intersection_20201008-2241231602193283.2883132/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/pss/intersection_20201008-2356301602197790.8946025/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/pss/intersection_20201021-1922561603304576.0938213/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/flow/intersection_20201021-1919381603304378.5587554/logs/train_log.json'
LOG_PATH = 'data/experiments/intbins/queue/intersection_20201009-1552361602255156.0307872/logs/train_log.json'

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

    parser.add_argument('--percentiles', type=float, nargs="+", required=False,
                    help='Percentiles array which values range from 0.01 until 0.99', default=[0.15, 0.5, 0.85])

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/state_space_visualizer):')
    print('\tExperiment collection folder: {0}\n'.format(args.experiment_collection_folder))
    print('\tPercentiles: {0}\n'.format(args.percentiles))

def main():

    print('\nRUNNING analysis/state_space_visualizer.py\n')

    args = get_arguments()
    # print_arguments(args)

    if Path(args.experiment_collection_folder).suffix == '.gz':
        raise ValueError('Please uncompress folder first.')
    origin_path = Path(args.experiment_collection_folder)

    train_log_paths = list(p for p in origin_path.rglob('*train_log.json'))

    percentiles = args.percentiles
    # raise ValueError('num_samples argument should be <= than the number of training runs.')

    # Data extraction loop
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

    # Data aggregation loop
    # paginates all tuples consolidating same features.
    data1 = {}
    for k, v in data.items():
        network, features = k
        for feature in features:
            if (network, feature) not in data1:
                data1[(network, feature)] = defaultdict(list)
                sel = {k1: fn(v1, k1[1].index(feature))
                       for k1, v1 in data.items() if k1[0] == network and feature in k1[1]}
                for cycles in sel.values():
                    for cycle in cycles:
                        for tl, phases in cycle.items():
                            data1[(network, feature)][tl].append(phases)

    tlids = sorted({k for dat in data1.values() for k in dat})
    networks = sorted({k[0] for dat in data1})

    # Data partitioning loop
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
                    data3[tid][feature] = {}
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

                    # ax.vlines(p1, 0, 1,
                    #           transform=ax.get_xaxis_transform(),
                    #           colors='tab:purple', label=f'phase 1')

                    # ax.hlines(p2, 0, 1,
                    #           transform=ax.get_yaxis_transform(),
                    #           colors='tab:cyan', label=f'phase 2')
                    plt.xlabel(f'{snakefy(feature)}')
                    plt.ylabel('Density')
                    plt.title(f'{snakefy(network)}: Intersection {tid}, {snakefy(feature)} feature')
                    plt.savefig((target_path / f'{tid}-{feature}.png').as_posix(),
                                bbox_inches='tight', pad_inches=0)
                    plt.close()

                # Save quantile plots
                df = pd.DataFrame.from_dict(data4).transpose()
                df.to_csv((target_path / 'quantile.csv').as_posix(), sep=',', encoding='utf-8')

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()


    # log_path = Path(ILU_PATH) / LOG_PATH
    # config_path = log_path.parent.parent / 'config'
    # quantile = (0.15, 0.5, 0.75, 0.85)
    # # TODO: go to config and determine features
    # train_config = configparser.ConfigParser()
    # train_config.read((config_path / 'train.config').as_posix())

    # # TODO: go to config and network
    # network = train_config.get('train_args', 'network')
    # features = eval(train_config.get('mdp_args', 'features'))

    # # 1. Create log_path
    # target_path = log_path.parent.parent / 'log_plots'
    # target_path.mkdir(mode=0o777, exist_ok=True)

    # # TODO: Aggregate multiple experiments
    # with log_path.open(mode='r') as f:

    #     json_data = json.load(f)

    #     states = json_data['states']
    #     states = pd.DataFrame(states)

    #     # Interates per intersection
    #     for (tid, tdata) in states.iteritems():

    #         # assumptions there are always two phases
    #         labels = [f'{label}_{n}' for n in range(2) for label in features]
    #         df = pd.DataFrame(tdata.to_list(), columns=labels)
    #         # Num. plots == Num. features  x Num. intersection
    #         # Num. series == Num. labels in feature
    #         for feature in features:
    #             # Delay.
    #             fig = plt.figure()
    #             fig.set_size_inches(FIGURE_X, FIGURE_Y)

    #             for num in range(2):
    #                 label = f'{feature}_{num}'
    #                 sns.distplot(df[label], hist=False, kde=True, label=label, kde_kws = {'linewidth': 3})

    #             plt.xlabel(f'{snakefy(feature)}')
    #             plt.ylabel('Density')
    #             plt.title(f'{snakefy(network)}: Intersection {tid}, {snakefy(feature)} feature')
    #             plt.savefig((target_path / f'{tid}-{feature}.png').as_posix(),
    #                         bbox_inches='tight', pad_inches=0)
    #             plt.close()

    #     # Save quantile plots
    #     qdf = df.quantile(quantile)
    #     print(qdf.head(len(quantile)))
    #     qdf.to_csv((target_path / 'quantile.csv').as_posix(), sep='|', encoding='utf-8')
