"""
    jobs/train.py
"""
from pathlib import Path
from datetime import datetime
import sys
import os
import time
import json
import tempfile
import configparser
import multiprocessing
import multiprocessing.pool

from ilurl.loaders.parser import config_parser
from models.train import main as train
from ilurl.utils.decorators import processable, benchmarked

ILURL_HOME = os.environ['ILURL_HOME']
CONFIG_PATH = \
    f'{ILURL_HOME}/config/'

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


@benchmarked
def benchmarked_train(*args, **kwargs):
    return train(*args, **kwargs)


def delay_train(args):
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
    return benchmarked_train(args[1])


def train_batch(categories={}):

    print('\nRUNNING jobs/train.py\n')
    num_processors, num_runs, train_seeds = config_parser.parse_run_params(print_params=False)


    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                        ' must match the number of runs (`num_runs`) argument.')

    print('\nArguments (jobs/train.py):')
    print('------------------------')
    print('Number of runs: {0}'.format(num_runs))
    print('Number of processors: {0}'.format(num_processors))
    print('Train seeds: {0}\n'.format(train_seeds))

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'WARNING: Number of processors downgraded to {num_processors}\n')

    # Read train.py arguments from train.config file.
    train_config = configparser.ConfigParser()
    train_config.read(os.path.join(CONFIG_PATH, 'train.config'))

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
    print(f'Experiment timestamp: {timestamp}\n')

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Create a config file for each train.py
        # with the respective seed. These config
        # files are stored in a temporary directory.
        train_configs = []
        for seed in train_seeds:

            tmp_train_cfg_path = os.path.join(tmp_dir,
                                        'train-{0}.config'.format(seed))
            train_configs.append(tmp_train_cfg_path)

            # Setup train seed.
            train_config.set("train_args", "experiment_seed", str(seed))
            if any(categories):
                train_config.set("mdp_args", "category", str(categories))

            # Write temporary train config file.
            tmp_cfg_file = open(tmp_train_cfg_path, "w")
            train_config.write(tmp_cfg_file)
            tmp_cfg_file.close()

        # Run.
        # rvs: directories' names holding experiment data
        if num_processors > 1:
            train_args = zip(range(num_runs), train_configs)
            pool = NonDaemonicPool(num_processors, maxtasksperchild=1)
            rvs = pool.map(delay_train, train_args)
            pool.close()
            pool.join()
        else:
            rvs = []
            for cfg in train_configs:
                rvs.append(delay_train((0.0, cfg)))

        # Create a directory and move newly created files.
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
def train_job():
    # Suppress textual output.
    return train_batch()

if __name__ == '__main__':
    train_batch() # Use this line for textual output.
    # train_job()
