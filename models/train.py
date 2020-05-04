"""Provides baseline for networks

    References:
    ==========
    * seed:
    https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
    http://sumo.sourceforge.net/userdoc/Simulation/Randomness.html
"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'
import json
from os import environ
from pathlib import Path

import numpy as np
import random

from flow.core.params import EnvParams, SumoParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS

from ilurl.core.experiment import Experiment
from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network

from ilurl.loaders.parsers import parse_train_params

ILURL_PATH = Path(environ['ILURL_HOME'])
EMISSION_PATH = ILURL_PATH / 'data/emissions/'
# CONFIG_PATH = ILURL_PATH / 'config'

def main(train_config_path=None):

    train_args = parse_train_params(train_config_path, print_params=True)

    inflows_type = 'switch' if train_args.inflows_switch else 'lane'
    network_args = {
        'network_id': train_args.network,
        'horizon': train_args.experiment_time,
        'demand_type': inflows_type,
        'insertion_probability': 0.1, # TODO: this needs to be network dependant
        'tls_type': train_args.tls_type
    }

    network = Network(**network_args)

    # Create directory to store data.
    experiment_path = EMISSION_PATH / network.name
    if not experiment_path.exists():
        experiment_path.mkdir()
    print(f'Experiment: {str(experiment_path)}')

    sumo_args = {
        'render': train_args.sumo_render,
        'print_warnings': False,
        'sim_step': 1, # step = 1 by default.
        'restart_instance': True
    }

    # Setup seeds.
    if train_args.experiment_seed is not None:
        random.seed(train_args.experiment_seed)
        np.random.seed(train_args.experiment_seed)
        sumo_args['seed'] = train_args.experiment_seed

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

    env = TrafficLightEnv(
        env_params=env_params,
        sim_params=sim_params,
        network=network,
        )

    # Override possible inconsistent params.
    if train_args.tls_type not in ('controlled',):
        env.stop = True
        train_args.save_agent = False
        train_args.save_agent_interval = None

    exp = Experiment(
            env=env,
            exp_path=experiment_path.as_posix(),
            train=True,
            log_info=train_args.experiment_log,
            log_info_interval=train_args.experiment_log_interval,
            save_agent=train_args.experiment_save_agent,
            save_agent_interval=train_args.experiment_save_agent_interval
     )

    # Store parameters.
    parameters = {}
    parameters['network_args'] = network_args
    parameters['sumo_args'] = sumo_args
    parameters['env_args'] = env_args
    params_path = experiment_path / "params.json" 
    with params_path.open('w') as f:
        json.dump(parameters, f)

    # Run the experiment.
    exp.run(train_args.experiment_time)

    return str(experiment_path)

if __name__ == '__main__':
    main()