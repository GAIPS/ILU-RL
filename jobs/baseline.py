"""Baseline:

    * uses train script for setting up experiment
    * has to have a command line `tls_type`: `actuated`, `static`


TODO: Include config for evaluation

"""
import os
from pathlib import Path
from datetime import datetime
import sys
import json
import tempfile
import multiprocessing
import multiprocessing.pool
import time
import shutil
import argparse

import configparser

from models.train import main as baseline
from ilurl.utils.decorators import processable, benchmarked
from ilurl.utils import str2bool

# Pipeline components.
from jobs.convert2csv import xml2csv
from analysis.test_plots import main as test_plots
from ilurl.utils.decorators import safe_run
_ERROR_MESSAGE_TEST = ("ERROR: Caught an exception while "
                    "executing analysis/test_plots.py script.")
test_plots = safe_run(test_plots, error_message=_ERROR_MESSAGE_TEST)


ILURL_PATH = Path(os.environ['ALTRL_HOME'])
CONFIG_PATH = ILURL_PATH / 'config'

mp = multiprocessing.get_context('spawn')


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


def delay_baseline(args):
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
    return baseline(args[1])


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            Script to evaluate actuated, static or random timings.
        """
    )

    parser.add_argument('tls_type', type=str, nargs='?',
                        choices=('actuated', 'static', 'webster', 'random', 'max_pressure'),
                         help='Control type.')
    flags = parser.parse_args()
    sys.argv = [sys.argv[0]]
    return flags

def baseline_batch():

    flags = get_arguments()

    # Read script arguments from run.config file.
    run_config = configparser.ConfigParser()
    run_path = CONFIG_PATH / 'run.config'
    run_config.read(run_path)

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    seeds = json.loads(run_config.get("run_args", "train_seeds"))

    if len(seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `seeds`'
                        ' must match the number of runs (`num_runs`) argument.')

    print('Arguments (baseline.py):')
    print('-----------------------')
    print('Number of runs: {0}'.format(num_runs))
    print('Number of processors: {0}'.format(num_processors))
    print('Train seeds: {0}\n'.format(seeds))

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'Number of processors downgraded to {num_processors}\n')

    # Read train.py arguments from train.config file.
    baseline_config = configparser.ConfigParser()
    baseline_path = CONFIG_PATH / 'train.config'
    baseline_config.read(str(baseline_path))
    
    # Setup sumo-tls-type.
    baseline_config.set('train_args', 'tls_type', flags.tls_type)
    baseline_config.set('train_args', 'experiment_save_agent', str(False))

    # Override train configurations with test parameters.
    test_config = configparser.ConfigParser()
    test_path = CONFIG_PATH / 'test.config'
    test_config.read(test_path.as_posix())

    horizon = int(test_config.get('test_args', 'rollout-time'))
    baseline_config.set('train_args', 'experiment_time', str(horizon))

    # Write .xml files for test plots creation.
    baseline_config.set('train_args', 'sumo_emission', str(True))

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
    print(f'Experiment timestamp: {timestamp}')

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a config file for each train.py
        # with the respective seed. These config
        # files are stored in a temporary directory.
        tmp_path = Path(tmp_dir)
        baseline_configs = []
        for seed in seeds:

            cfg_path = tmp_path / f'{flags.tls_type}-{seed}.config'
            baseline_configs.append(cfg_path)

            # Setup train seed.
            baseline_config.set("train_args", "experiment_seed", str(seed + 1))
            
            # Write temporary train config file.
            with cfg_path.open('w') as ft:
                baseline_config.write(ft)

        # rvs: directories' names holding experiment data
        if num_processors > 1:
            packed_args = [(delay, cfg)
                                for (delay, cfg) in zip(range(len(baseline_configs)), baseline_configs)]
            pool = NonDaemonicPool(num_processors)
            rvs = pool.map(delay_baseline, packed_args)
            pool.close()
            pool.join()
        else:
            rvs = []
            for cfg in baseline_configs:
                rvs.append(delay_baseline((0.0, cfg)))

        # Create a directory and move newly created files
        paths = [Path(f) for f in rvs]
        commons = [p.parent for p in paths]
        if len(set(commons)) > 1:
            raise ValueError(f'Directories {set(commons)} must have the same root')
        dirpath = commons[0]
        batchpath = dirpath / timestamp
        if not batchpath.exists():
            batchpath.mkdir()

        # Move files
        for src in paths:
            dst = batchpath / src.parts[-1]
            src.replace(dst)

    sys.stdout.write(str(batchpath))

    return str(batchpath)

@processable
def baseline_job():
    # Suppress textual output.
    return baseline_batch()

if __name__ == '__main__':

    # 1) Run baseline.
    experiment_root_path = baseline_batch()

    # 2) Convert .xml files to .csv files.
    xml2csv(experiment_root_path=experiment_root_path)

    # 3) Create plots with metrics plots for final agent.
    test_plots(experiment_root_path)

    # 4) Clean up and compress files in order
    #    to optimize disk space.
    print('\nCleaning and compressing files...\n')
    experiment_root_path = Path(experiment_root_path)
    for csv_path in experiment_root_path.rglob('*-emission.csv'):
        Path(csv_path).unlink()

    shutil.make_archive(experiment_root_path,
                    'gztar',
                    os.path.dirname(experiment_root_path),
                    experiment_root_path.name)
    
    shutil.rmtree(experiment_root_path)

    print('Experiment folder: {0}'.format(experiment_root_path))
