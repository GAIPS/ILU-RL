"""
    models/train.py

    This script trains a multi-agent system under a given network and demand type.
    It is responsible to setup all the experiment's components and run it for
    a given number of steps, as well as storing experiment-related info.

    References:
    ==========

    * Seeds:
    https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
    http://sumo.sourceforge.net/userdoc/Simulation/Randomness.html

"""
import os
import json
import random
from pathlib import Path
from shutil import copyfile

import numpy as np

from flow.core.params import EnvParams, SumoParams

from ilurl.experiment import Experiment
from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network

from ilurl.loaders.parser import config_parser


def main(preprocess_config_path=None):

    # Setup config parser path.
    if preprocess_config_path is not None:
        print(f'Loading train parameters from: {preprocess_config_path}')
        config_parser.set_config_path(preprocess_config_path)
    else:
        print('Loading train parameters from: configs/train.config [DEFAULT]')

    # Parse pre-process parameters.
    # Since there is not a preprocess file we just use config/train.config
    preprocess_args = config_parser.parse_train_params(print_params=True)

    network_args = {
        'network_id': preprocess_args.network,
        'horizon': preprocess_args.experiment_time,
        'demand_type': preprocess_args.demand_type,
        'demand_mode': preprocess_args.demand_mode,
        'tls_type': preprocess_args.tls_type
    }
    network = Network(**network_args)

    # Create directory to store data.
    # experiment_path = EMISSION_PATH / network.name
    # os.makedirs(experiment_path, exist_ok=True)
    # print(f'Experiment: {str(experiment_path)}\n')

    sumo_args = {
        'render': preprocess_args.sumo_render,
        'print_warnings': False,
        'sim_step': 1, # Do not change.
        'restart_instance': True,
        'teleport_time': 120
    }

    # Setup seeds.
    if preprocess_args.experiment_seed is not None:
        random.seed(preprocess_args.experiment_seed)
        np.random.seed(preprocess_args.experiment_seed)
        sumo_args['seed'] = preprocess_args.experiment_seed

    sim_params = SumoParams(**sumo_args)

    additional_params = {}
    additional_params['tls_type'] = preprocess_args.tls_type
    env_args = {
        'evaluate': True,
        'additional_params': additional_params
    }
    env_params = EnvParams(**env_args)

    # Load MDP parameters from file (train.config[mdg_args]).
    mdp_params = config_parser.parse_mdp_params(network.tls_ids)

    env = TrafficLightEnv(
        env_params=env_params,
        sim_params=sim_params,
        mdp_params=mdp_params,
        network=network,
        exp_path=None,
        seed=preprocess_args.experiment_seed,
    )

    exp = Experiment(
            env=env,
            train=True,
            exp_path=None,
            save_agent=preprocess_args.experiment_save_agent,
            save_agent_interval=preprocess_args.experiment_save_agent_interval,
            tls_type=preprocess_args.tls_type
    )

    # Run the experiment.
    info_dict = exp.run(preprocess_args.experiment_time)

    return info_dict

if __name__ == '__main__':
    main()
