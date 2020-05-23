"""
    models/rollout.py

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

    parser.add_argument('--run-path', '-p', dest='run_path' , type=str, nargs='?',
                        help='Train run name to use for evaluation.')

    parser.add_argument('--chkpt-number', '-n', dest='chkpt_number', type=int,
                        nargs='?', required=True, help='Checkpoint number.')

    parser.add_argument('--rollout-time', '-t', dest='rollout_time', type=int,
                        default=300, nargs='?',
                        help='Experiment runtime to perform evaluation.')

    parser.add_argument('--sumo-emission', '-e', dest='sumo_emission', type=str2bool,
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

    print('Arguments (rollout.py):')
    print('----------------------')
    print('Run path: {0}'.format(args.run_path))
    print('Checkpoint number: {0}'.format(args.chkpt_number))
    print('Experiment time: {0}'.format(args.rollout_time))
    print('SUMO emission: {0}'.format(args.sumo_emission))
    print('SUMO render: {0}'.format(args.sumo_render))
    print('Seed: {0}\n'.format(args.seed))


def main(config_file_path=None):

    args = get_arguments(config_file_path)
    print_arguments(args)

    print('Loading from train run: {0}\n'.format(args.run_path))

    # Setup parser with custom path (load correct train parameters).
    config_path = Path(args.run_path) / 'config' / 'train.config'
    config_parser.set_config_path(config_path)

    # Parse train parameters.
    train_args = config_parser.parse_train_params(print_params=False)

    network_args = {
        'network_id': train_args.network,
        'horizon': args.rollout_time,
        'demand_type': train_args.demand_type,
        'tls_type': train_args.tls_type
    }

    network = Network(**network_args)

    # Create directory to store evaluation script data.
    experiment_path = Path(args.run_path) / 'eval' / f'{network.name}'
    os.makedirs(experiment_path, exist_ok=True)
    print(f'Experiment: {str(experiment_path)}')
    
    sumo_args = {
        'render': args.sumo_render,
        'print_warnings': False,
        'sim_step': 1,
        'restart_instance': True,
        'teleport_time': 180
    }

    # Setup seeds.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        sumo_args['seed'] = args.seed

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
    checkpoints_dir_path = Path(args.run_path) / 'checkpoints'
    env.agents.load_checkpoint(checkpoints_dir_path, args.chkpt_number)

    # Stop training.
    env.stop = True

    exp = Experiment(
            env=env,
            exp_path=experiment_path.as_posix(),
            train=False, # Stop training.
    )

    # Run the experiment.
    info_dict = exp.run(args.rollout_time)

    return info_dict

if __name__ == '__main__':
    main()