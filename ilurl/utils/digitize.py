"""This script helps setting sensible discretization bins

    This script performs the digitalization from continous spaces to discrete bins.
        * Splits results per network, tls and phase.
        * It searches for both train_logs and train_config for each experiment.
        * Aggregating MDPs when necessary for enrichment of datasets.
"""
import re
from os import environ
from pathlib import Path
import json
from collections import defaultdict
import numpy as np

ILURL_PATH = environ['ILURL_HOME']
MATCHER = re.compile('\[(.*?)\]')
NUM_PHASES = 2  #TODO: RELAX THIS ASSUMPTION
import configparser

def fn(cycles, ind):
    return [
                {
                    tid: [phase[ind] for phase in phases]
                    for tid, phases in cycle.items()
                }
                for cycle in cycles
            ]
def rmlag(x):
    if 'lag[' in x:
        return MATCHER.search(x).groups()[0]
    return x

def digitize(train_log_paths, percentiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Compute discretization bins.

    Params:
        * train_log_paths: list<pathlib.Path>
           A list of paths which point to root directory wrt to training logs.

        * percentiles: list<float>
           A list of normalized (0,1) increasing percentiles.

    Return:
        * categories: dict<str,dict<int, list<float>>>
            tlid x phases x scores

        * networks: list
            list of networks
    """
    data = _loader(train_log_paths)

    data, networks = _consolidate(data)

    categories = _apply(data, percentiles)

    return categories, networks

def digitize2(train_logs, percentiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Compute discretization bins from data
    Params:
        * train_logs: dict<tid, dict<>>
           A list of paths which point to root directory wrt to training logs.

        * percentiles: list<float>
           A list of normalized (0,1) increasing percentiles.

    Return:
        * categories: dict<str,dict<int, list<float>>>
            tlid x phases x scores

        * networks: list
            list of networks
    """
    data, _ = _consolidate(train_logs)

    categories = _apply(data, percentiles)

    return categories

def _consolidate(data, verbose=True):
    # Data aggregation loop
    # paginates all tuples consolidating same features.
    # i.e 'count' from mdp speed_delta and speed_score are aggregated
    # i.e 'lag[.]' states are excluded
    data1 = {}
    if verbose:
        print('Merging experiments...')

    for k, v in data.items():
        network, features = k
        for feature in features:
            if (network, feature) not in data1:
                data1[(network, feature)] = defaultdict(list)
                sel = {k1: fn(v1, k1[1].index(feature))
                       for k1, v1 in data.items() if k1[0] == network and feature in k1[1]}

                num_points = []
                for cycles in sel.values():
                    for cycle in cycles:
                        for tl, phases in cycle.items():
                            data1[(network, feature)][tl].append(phases)
                            num_points.append(len(phases))
                if verbose:
                    msgs = [f'{network}\tnum. nodes: {len(data1[(network, feature)])}']
                    msgs.append(f'feature: {feature}')
                    msgs.append(f'num. data points:{sum(num_points)}')
                    print("\t".join(msgs))
    networks = sorted({k[0] for k in data})
    return data1, networks

def _loader(train_log_paths):
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

    return data

def _apply(data, percentiles):
    """Segregating results per network applies percentiles"""

    networks = sorted({k[0] for k in data})
    data2 = {}
    for network in networks:
        data1 = {k: v for k, v in data.items() if network == k[0]}
        tlids = sorted({kk for dat in data1.values() for kk in dat})

        for tid in tlids:
            data2[tid] = {}
            for key, values in data1.items():
                _, feature = key
                d1, d2 = zip(*values[tid]) # should have two phases
                p1 = np.quantile(d1, percentiles).tolist()
                p2 = np.quantile(d2, percentiles).tolist()
                data2[tid][feature] = {'0': p1, '1': p2}
    return data2
