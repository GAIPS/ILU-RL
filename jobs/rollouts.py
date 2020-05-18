"""
    jobs/rollouts.py
"""

from pathlib import Path
import itertools
import sys
from os import environ
import json
import tempfile
import argparse
import multiprocessing as mp
import time
from collections import defaultdict

import configparser

from ilurl.utils.decorators import processable
from models.rollout import main as roll

ILURL_HOME = environ['ILURL_HOME']
CONFIG_PATH = Path(f'{ILURL_HOME}/config/')


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This scripts runs recursively a rollout for every checkpoint stored
            on the experiment path. If test is set to True only the last checkpoints
            will be used.
        """
    )
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        help='''A directory which it\'s subdirectories are train runs.''')

    parser.add_argument('--test', '-t', dest='test', type=str2bool,
                        default=False, nargs='?',
                        help='If true only the last checkpoints will be used.')

    parsed = parser.parse_args()
    sys.argv = [sys.argv[0]]

    return parsed

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def concat(evaluations):
    """Receives an experiments' json and merges it's contents

    Params:
    -------
        * evaluations: list
        list of rollout evaluations

    Returns:
    --------
        * result: dict
        where `id` key certifies that experiments are the same
              `list` params are united
              `numeric` params are appended

    """
    result = defaultdict(list)
    for qtb in evaluations:
        exid = qtb.pop('id')
        qid = qtb.get('rollouts', 0)[0]
        # can either be a rollout from the prev
        # exid or a new experiment
        if exid not in result['id']:
            result['id'].append(exid)

        ex_idx = result['id'].index(exid)
        for k, v in qtb.items():
            append = isinstance(v, list) or isinstance(v, dict)
            # check if integer fields match
            # such as cycle, save_step, etc
            if not append:
                if k in result:
                    if result[k] != v:
                        raise ValueError(
                            f'key:\t{k}\t{result[k]} and {v} should match'
                        )
                else:
                    result[k] = v
            else:
                if ex_idx == len(result[k]):
                    result[k].append(defaultdict(list))
                result[k][ex_idx][qid].append(v)
    return result


def rollout_batch(test=False, experiment_dir=None):

    print('\nRUNNING jobs/rollouts.py\n')

    if not experiment_dir:

        # Read script arguments.
        args = get_arguments()
        # Clear command line arguments after parsing.

        batch_path = Path(args.experiment_dir)
        test = args.test

    else:
        batch_path = Path(experiment_dir)

    chkpt_pattern = '*.chkpt'

    # Get names of train runs.
    experiment_names = list({p.parents[1] for p in batch_path.rglob(chkpt_pattern)})

    # Get checkpoints numbers.
    chkpts_nums = list({int(p.stem.split('-')[1]) for p in batch_path.rglob(chkpt_pattern)})

    # If test then pick only the last checkpoints.
    if test:

        chkpts_nums = [chkpts_nums[-1]]
        rollouts_paths = list(itertools.product(experiment_names, chkpts_nums))
    
        print('\tjobs/rollouts.py (test mode): using checkpoints'
                ' number {0}'.format(chkpts_nums[0]))
    else:
        rollouts_paths = list(itertools.product(experiment_names, chkpts_nums))

    run_config = configparser.ConfigParser()
    run_config.read(str(CONFIG_PATH / 'run.config'))

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    train_seeds = json.loads(run_config.get("run_args", "train_seeds"))

    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                        'must match the number of runs (`num_runs`) argument.')

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'Number of processors downgraded to {num_processors}\n')

    # Read rollouts arguments from rollouts.config file.
    rollouts_config = configparser.ConfigParser()
    rollouts_config.read(str(CONFIG_PATH / 'rollouts.config'))
    num_rollouts = int(rollouts_config.get('rollouts_args', 'num-rollouts'))
    rollouts_config.remove_option('rollouts_args', 'num-rollouts')

    if test:
        # Merge test and rollouts config files.
        test_config = configparser.ConfigParser()
        test_config.read(str(CONFIG_PATH / 'test.config'))

        # By default in test mode only one rollout per checkpoint is done.
        num_rollouts = 1

        rollout_time = test_config.get('test_args', 'rollout-time')
        emission = test_config.get('test_args', 'sumo-emission')
        seed_delta = int(test_config.get('test_args', 'seed-delta'))

        # Overwrite defaults.
        rollouts_config.set('rollouts_args', 'rollout-time', rollout_time)
        rollouts_config.set('rollouts_args', 'sumo-emission', emission)

        # Alocate the S seeds among M rollouts.
        custom_configs = []
        base_seed = max(train_seeds)
        for rn, rp in enumerate(rollouts_paths):
            custom_configs.append((rp, base_seed + rn + seed_delta))
        token = 'test'

    else:
        # number of processes vs layouts
        # seeds must be different from training
        custom_configs = []
        for rn, rp in enumerate(rollouts_paths):
            base_seed = max(train_seeds) + num_rollouts * rn
            for rr in range(num_rollouts):
                seed = base_seed + rr + 1
                custom_configs.append((rp, seed))
        token = 'rollouts'

    # print(custom_configs)

    print(f'\nArguments (jobs/{token}.py):')
    print('-------------------------')
    print(f'Experiment dir: {batch_path}')
    print(f'Number of processors: {num_processors}')
    print(f'Num. rollout files: {len(rollouts_paths)}')
    print(f'Num. rollout repetitions: {num_rollouts}')
    print(f'Num. rollout total: {len(rollouts_paths) * num_rollouts}')
    print(f'Rollouts time: {rollout_time}')
    print(f'Rollouts emission: {emission}\n')

    with tempfile.TemporaryDirectory() as f:

        tmp_path = Path(f)
        # Create a config file for each rollout
        # with the respective seed. These config
        # files are stored in a temporary directory.
        rollouts_cfg_paths = []
        cfg_key = "rollouts_args"
        for cfg in custom_configs:
            run_path, chkpt_num = cfg[0]
            seed = cfg[1]

            # Setup custom rollout settings.
            rollouts_config.set(cfg_key, "run-path", str(run_path))
            rollouts_config.set(cfg_key, "chkpt-number", str(chkpt_num))
            rollouts_config.set(cfg_key, "seed", str(seed))
            
            # Write temporary train config file.
            cfg_path = tmp_path / f'rollouts-{run_path.name}-{chkpt_num}-{seed}.config'
            rollouts_cfg_paths.append(str(cfg_path))
            with cfg_path.open('w') as fw:
                rollouts_config.write(fw)

        # rvs: directories' names holding experiment data
        if num_processors > 1:
            pool = mp.Pool(num_processors)
            rvs = pool.map(roll, [[cfg] for cfg in rollouts_cfg_paths])
            pool.close()
        else:
            rvs = []
            for cfg in rollouts_cfg_paths:
                rvs.append(roll([cfg]))

    """ res = concat(rvs)
    print(res)
    res['num_rollouts'] = num_rollouts
    filepart = 'test' if test else 'eval'
    filename = f'{batch_path.parts[-1]}.l.{filepart}.info.json'
    target_path = batch_path / filename
    with target_path.open('w') as fj:
        json.dump(res, fj)

    sys.stdout.write(str(batch_path))
    return str(batch_path) """

@processable
def rollout_job(test=False):
    # Suppress textual output.
    return rollout_batch(test=test)

if __name__ == '__main__':
    #rollout_job()
    rollout_batch() # Use this line for textual output.
