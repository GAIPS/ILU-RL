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


ALTRL_PATH = Path(os.environ['ALTRL_HOME'])
EMISSION_PATH = ALTRL_PATH / 'data/emissions/'
NETWORKS_PATH = ALTRL_PATH / 'data/networks/'

def main(train_config_path=None):

    # Setup config parser path.
    if train_config_path is not None:
        print(f'Loading train parameters from: {train_config_path}')
        config_parser.set_config_path(train_config_path)
    else:
        print('Loading train parameters from: configs/train.config [DEFAULT]')

    # Parse train parameters.
    train_args = config_parser.parse_train_params(print_params=True)

    network_args = {
        'network_id': train_args.network,
        'horizon': train_args.experiment_time,
        'demand_type': train_args.demand_type,
        'demand_mode': train_args.demand_mode,
        'tls_type': train_args.tls_type
    }
    network = Network(**network_args)

    # Create directory to store data.
    experiment_path = EMISSION_PATH / network.name
    os.makedirs(experiment_path, exist_ok=True)
    print(f'Experiment: {str(experiment_path)}\n')

    sumo_args = {
        'render': train_args.sumo_render,
        'print_warnings': False,
        'sim_step': 1, # Do not change.
        'restart_instance': True,
        'teleport_time': 120
    }

    # Setup seeds.
    if train_args.experiment_seed is not None:
        random.seed(train_args.experiment_seed)
        np.random.seed(train_args.experiment_seed)
        sumo_args['seed'] = train_args.experiment_seed

    # Setup emission path.
    if train_args.sumo_emission:
        sumo_args['emission_path'] = experiment_path.as_posix()

    sim_params = SumoParams(**sumo_args)

    additional_params = {}
    additional_params['tls_type'] = train_args.tls_type
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
        exp_path=experiment_path.as_posix(),
        seed=train_args.experiment_seed,
    )

    exp = Experiment(
            env=env,
            exp_path=experiment_path.as_posix(),
            train=True,
            save_agent=train_args.experiment_save_agent,
            save_agent_interval=train_args.experiment_save_agent_interval,
            tls_type=train_args.tls_type
    )

    # Store train parameters (config file). 
    config_parser.store_config(experiment_path / 'config')

    # Store a copy of the tls_config.json file.
    tls_config_path = NETWORKS_PATH / train_args.network / 'tls_config.json'
    copyfile(tls_config_path, experiment_path / 'tls_config.json')

    # Store a copy of the demands.json file.
    demands_file_path = NETWORKS_PATH / train_args.network / 'demands.json'
    copyfile(demands_file_path, experiment_path / 'demands.json')

    # Run the experiment.
    info_dict = exp.run(train_args.experiment_time)

    # Store train info dict.
    os.makedirs(experiment_path/ 'logs', exist_ok=True)
    train_log_path = experiment_path / 'logs' / "train_log.json"
    with train_log_path.open('w') as f:
        json.dump(info_dict, f)

    return str(experiment_path)

if __name__ == '__main__':
    main()
