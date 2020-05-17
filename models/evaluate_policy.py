"""
    models/evaluate_policy.py

    This script evaluates a given RL policy.
    It performs a rollout given a static policy
    loaded from a given model checkpoint.
"""
import os
import json
import argparse
import pickle
from pathlib import Path

import configargparse
import numpy as np
import random

from flow.core.params import SumoParams, EnvParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS

from ilurl.core.experiment import Experiment
from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network

from ilurl.loaders.parser import config_parser


ILURL_PATH = Path(os.environ['ILURL_HOME'])
EMISSION_PATH = ILURL_PATH / 'data/emissions/'

def get_arguments(config_file_path):
    if config_file_path is None:
        config_file_path = []

    parser = configargparse.ArgumentParser(
        default_config_files=config_file_path,
        description="""
            This script evaluates a given RL policy.
            It performs a rollout given a static policy
            loaded from a given model checkpoint.
        """
    )

    parser.add_argument('train_run_path', type=str, nargs='?',
                        help='Train run name to use for evaluation.')

    parser.add_argument('--chckpt-number', '-n', dest='chkpt_number', type=int,
                        nargs='?', required=True, help='Checkpoint number.')

    parser.add_argument('--time', '-t', dest='exp_time', type=int,
                        default=300, nargs='?',
                        help='Experiment time to perform evaluation.')

    parser.add_argument('--emission', '-e', dest='sumo_emission', type=str2bool,
                        default=True, nargs='?',
                        help='Enabled will perform saves')

    parser.add_argument('--sumo-render', '-r', dest='sumo_render', type=str2bool,
                        default=False, nargs='?',
                        help='If true renders the simulation.')

    parser.add_argument('--seed', '-d', dest='seed', type=int,
                        default=None, nargs='?',
                        help='''Sets seed value for both rl agent and Sumo.
                               `None` for rl agent defaults to RandomState() 
                               `None` for Sumo defaults to a fixed but arbitrary seed''')

    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_arguments(args):

    print('Arguments (evaluate_policy.py):')
    print('\tTrain run path: {0}'.format(args.train_run_path))
    print('\tCheckpoint number: {0}'.format(args.chkpt_number))
    print('\tExperiment time: {0}'.format(args.exp_time))
    print('\tSUMO emission: {0}'.format(args.sumo_emission))
    print('\tSUMO render: {0}'.format(args.sumo_render))
    print('\tSeed: {0}\n'.format(args.seed))

def setup_programs(programs_json):
    programs = {}
    for tls_id in programs_json.keys():
        programs[tls_id] = {int(action): programs_json[tls_id][action]
                                for action in programs_json[tls_id].keys()}
    return programs


def main(config_file_path=None):

    args = get_arguments(config_file_path)
    print_arguments(args)

    print('Loading from train run: {0}\n'.format(args.train_run_path))

    # Setup parser with custom path (load correct train parameters).
    config_path = Path(args.train_run_path) / 'config' / 'train.config'
    print(config_path)
    config_parser.set_config_path(config_path)

    # Parse train parameters.
    train_args = config_parser.parse_train_params(print_params=False)

    network_args = {
        'network_id': train_args.network,
        'horizon': args.exp_time,
        'demand_type': train_args.demand_type,
        'tls_type': train_args.tls_type
    }

    network = Network(**network_args)

    # Create directory to store evaluation script data.
    experiment_path = Path(args.train_run_path) / 'eval' / f'{network.name}'
    os.makedirs(experiment_path, exist_ok=True)
    print(f'Experiment: {str(experiment_path)}')
    
    sumo_args = {
        'render': args.sumo_render,
        'print_warnings': False,
        'sim_step': 1,
        'restart_instance': True
    }

    # Setup seeds.
    if args.seed is not None:
        random.seed(args.experiment_seed)
        np.random.seed(args.experiment_seed)
        sumo_args['seed'] = args.experiment_seed

    # Setup emission path.
    if args.sumo_emission:
        sumo_args['emission_path'] = experiment_path.as_posix()

    sim_params = SumoParams(**sumo_args)

    additional_params = {}
    additional_params.update(ADDITIONAL_ENV_PARAMS)
    additional_params['tls_type'] = train_args.tls_type
    env_args = {
        'evaluate': True,
        'additional_params': additional_params
    }
    env_params = EnvParams(**env_args)

    env = TrafficLightEnv(
        env_params=env_params,
        sim_params=sim_params,
        network=network,
        )

    # Setup checkpoints.
    checkpoints_dir_path = Path(args.train_run_path) / 'checkpoints'
    env.agents.load_checkpoint(checkpoints_dir_path, args.chkpt_number)

    # Stop training.
    env.stop = True

    exp = Experiment(
            env=env,
            exp_path=experiment_path.as_posix(),
            train=False,
    )

    # Run the experiment.
    exp.run(args.exp_time)

if __name__ == '__main__':
    main()