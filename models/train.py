"""
    models/train.py

    References:
    ==========

    * Seeds:
    https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
    http://sumo.sourceforge.net/userdoc/Simulation/Randomness.html

"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'
import os
import json
import random
from pathlib import Path
from shutil import copyfile

import numpy as np

from flow.core.params import EnvParams, SumoParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS

from ilurl.core.experiment import Experiment
from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network

from ilurl.loaders.parser import config_parser

from baselines.common import set_global_seeds

ILURL_PATH = Path(os.environ['ILURL_HOME'])
EMISSION_PATH = ILURL_PATH / 'data/emissions/'
NETWORKS_PATH = ILURL_PATH / 'data/networks/'

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
        'sim_step': 1,
        'restart_instance': True,
        'teleport_time': 180
    }

    # Setup seeds.
    if train_args.experiment_seed is not None:
        random.seed(train_args.experiment_seed)
        np.random.seed(train_args.experiment_seed)
        sumo_args['seed'] = train_args.experiment_seed
        set_global_seeds(train_args.experiment_seed)

    # Setup emission path.
    if train_args.sumo_emission:
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


    # Load MDP parameters from file (train.config[mdg_args]).
    mdp_params = config_parser.parse_mdp_params()

    env = TrafficLightEnv(
        env_params=env_params,
        sim_params=sim_params,
        mdp_params=mdp_params,
        network=network,
        )

    # Override possible inconsistent params.
    if train_args.tls_type not in ('controlled',):
        env.stop = True
        train_args.save_agent = False

    exp = Experiment(
            env=env,
            exp_path=experiment_path.as_posix(),
            train=True,
            log_info=train_args.experiment_log,
            log_info_interval=train_args.experiment_log_interval,
            save_agent=train_args.experiment_save_agent,
            save_agent_interval=train_args.experiment_save_agent_interval
    )

    # Store train parameters (config file). 
    config_parser.store_config(experiment_path / 'config')

    # Store a copy of the tls_config.json file.
    tls_config_path = NETWORKS_PATH / train_args.network / 'tls_config.json'
    copyfile(tls_config_path, experiment_path / 'tls_config.json')

    # Run the experiment.
    info_dict = exp.run(train_args.experiment_time)

    # Store train info dict.
    train_log_path = experiment_path / 'logs' / "train_log.json"
    with train_log_path.open('w') as f:
        json.dump(info_dict, f)

    return str(experiment_path)

if __name__ == '__main__':
    main()
