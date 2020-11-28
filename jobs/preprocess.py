"""Baseline:

    * uses train script for setting up experiment
    * has to have a command line `tls_type`: `actuated`, `static`


TODO: Include config for evaluation

"""
from collections import defaultdict
from os import environ
from pathlib import Path
from datetime import datetime
import json
import tempfile
import multiprocessing
import multiprocessing.pool
import time
import re

import configparser

from models.preprocess import main as preprocess

# Pipeline components.
from jobs.convert2csv import xml2csv
from analysis.baseline_plots import main as baseline_plots
from ilurl.utils.digitize import digitize2

ILURL_PATH = Path(environ['ILURL_HOME'])
CONFIG_PATH = ILURL_PATH / 'config'

mp = multiprocessing.get_context('spawn')

MATCHER = re.compile('\[(.*?)\]')

def do_preprocess():
    train_config = configparser.ConfigParser()
    train_path = CONFIG_PATH / 'train.config'
    train_config.read(train_path)

    return eval(train_config.get('mdp_args', 'discretize_state_space'))

def rmlag(x):
    if 'lag[' in x:
        return MATCHER.search(x).groups()[0]
    return x

class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass

class NoDaemonContext(type(multiprocessing.get_context('spawn'))):
    Process = NoDaemonProcess

class NonDaemonicPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NonDaemonicPool, self).__init__(*args, **kwargs)


def delay_preprocess(args):
    """Delays execution.

        Parameters:
        -----------
        * args: tuple
            Position 0: execution delay of the process.
            Position 1: store the train config file.

        Returns:
        -------
        * fnc : function
            An anonymous function to be executed with a given delay
    """
    time.sleep(args[0])
    return preprocess(*args[1:])


def preprocess_batch(tls_type='webster'):
    # Preprocess is a run with some presets
    # Read script arguments from run.config file.
    run_config = configparser.ConfigParser()
    run_path = CONFIG_PATH / 'run.config'
    run_config.read(run_path)

    data = {}
    if do_preprocess():
        num_processors = int(run_config.get('run_args', 'num_processors'))
        num_runs = int(run_config.get('run_args', 'num_runs'))
        seeds = json.loads(run_config.get("run_args", "train_seeds"))

        if len(seeds) != num_runs:
            raise configparser.Error('Number of seeds in run.config `seeds`'
                            ' must match the number of runs (`num_runs`) argument.')

        # Assess total number of processors.
        processors_total = mp.cpu_count()
        print(f'Total number of processors available: {processors_total}\n')

        # Adjust number of processors.
        if num_processors > processors_total:
            num_processors = processors_total
            print(f'Number of processors downgraded to {num_processors}\n')
        # num_processors should be <= num_runs
        seeds = seeds[:num_processors]

        print('Arguments (preprocess):')
        print('-----------------------')
        print(f'Number of runs: {num_runs}')
        print(f'Number of processors: {num_processors}')
        print(f'Number of preprocess trails: {num_processors}\n')


        # Read train.py arguments from train.config file.
        preprocess_config = configparser.ConfigParser()
        preprocess_path = CONFIG_PATH / 'train.config'
        preprocess_config.read(str(preprocess_path))

        # Setup sumo-tls-type.
        preprocess_config.set('train_args', 'tls_type', tls_type)
        preprocess_config.set('train_args', 'experiment_save_agent', str(False))
        preprocess_config.set('mdp_args', 'discretize_state_space', str(False))

        # Get feature & network information
        network = preprocess_config.get('train_args', 'network')
        features = eval(preprocess_config.get('mdp_args', 'features'))

        # Remove lag from features
        features = tuple(rmlag(f) for f in features)

        # Override train configurations with test parameters.
        test_config = configparser.ConfigParser()
        test_path = CONFIG_PATH / 'test.config'
        test_config.read(test_path.as_posix())

        horizon = int(test_config.get('test_args', 'rollout-time'))
        preprocess_config.set('train_args', 'experiment_time', str(horizon))

        # Write .xml files for test plots creation.
        preprocess_config.set('train_args', 'sumo_emission', str(False))

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
        print(f'Experiment timestamp: {timestamp}')

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a config file for each train.py
            # with the respective seed. These config
            # files are stored in a temporary directory.
            tmp_path = Path(tmp_dir)
            preprocess_configs = []
            for seed in seeds:

                cfg_path = tmp_path / f'{tls_type}-{seed}.config'
                preprocess_configs.append(cfg_path)

                # Setup train seed.
                preprocess_config.set("train_args", "experiment_seed", str(seed + 1))

                # Write temporary train config file.
                with cfg_path.open('w') as ft:
                    preprocess_config.write(ft)

            # rvs: directories' names holding experiment data
            if num_processors > 1:
                ind = range(num_processors)
                cfgs = preprocess_configs
                packed_args = zip(ind, cfgs)
                pool = NonDaemonicPool(num_processors)
                rvs = pool.map(delay_preprocess, packed_args)
                pool.close()
                pool.join()
            else:
                rvs = []
                for cfg in preprocess_configs:
                    rvs.append(delay_preprocess((0.0, cfg)))


        data = defaultdict(list)
        for ret in rvs:
            data[(network, features)] += ret['observation_spaces']
        data = digitize2(data)

    return data

if __name__ == '__main__':
    preprocess_batch()

