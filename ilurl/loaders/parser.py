import json
from os import environ
from pathlib import Path
from ast import literal_eval

from shutil import copyfile

import configparser

from ilurl.params import (QLParams,
                               DQNParams,
                               MDPParams,
                               TrainParams)

AGENT_TYPES = ('QL', 'DQN')

ILURL_PATH = Path(environ['ILURL_HOME'])
TRAIN_CONFIG_PATH = ILURL_PATH / 'config/train.config'

def str2bool(v, exception=None):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def isNone(string):
    if string in ('None', 'none'):
        return True
    else:
        return False

class Parser(object):
    """
        Parser for experiment parameters.
    """

    def __init__(self):
        self.config_path = TRAIN_CONFIG_PATH # Default config path.

    def set_config_path(self, config_path):
        """
            Overrides default config path.
        """
        self.config_path = config_path # Custom config path.

    def store_config(self, save_dir_path):
        """
            Stores config file in 'save_dir_path' directory.
        """
        if not save_dir_path.exists():
            save_dir_path.mkdir()
        copyfile(self.config_path, save_dir_path / 'train.config')

    def parse_train_params(self, print_params=False):
        """
            Parses train.py parameters (train_args) from config file located at
            self.config_path and returns a ilurl.core.params.TrainParams object
            with the parsed parameters.
        """
        # Load config file with parameters.
        train_config = configparser.ConfigParser()

        train_config.read(str(self.config_path))

        train_args = train_config['train_args']

        seed = int(train_args['experiment_seed']) if not isNone(train_args['experiment_seed']) else None

        train_params = TrainParams(
                        network=train_args['network'],
                        experiment_time=int(train_args['experiment_time']),
                        experiment_save_agent=str2bool(train_args['experiment_save_agent']),
                        experiment_save_agent_interval=int(train_args['experiment_save_agent_interval']),
                        experiment_seed=seed,
                        sumo_render=str2bool(train_args['sumo_render']),
                        sumo_emission=str2bool(train_args['sumo_emission']),
                        tls_type=train_args['tls_type'],
                        demand_type=train_args['demand_type'],
        )

        if print_params:
            print(train_params)

        return train_params


    def parse_mdp_params(self):
        """
            Parses MDP parameters (mdp_args) from config file located
            at self.config_path and returns a ilurl.core.params.MDPParams
            object with the parsed parameters.
        """
        # Load config file with parameters.
        train_config = configparser.ConfigParser()
        train_config.read(str(self.config_path))

        mdp_args = train_config['mdp_args']
        mdp_params = MDPParams(
            states=literal_eval(mdp_args['states']),
            category_counts=json.loads(mdp_args['category_counts']),
            category_delays=json.loads(mdp_args['category_delays']),
            category_speeds=json.loads(mdp_args['category_speeds']),
            category_queues=json.loads(mdp_args['category_queues']),
            normalize_state_space=str2bool(mdp_args['normalize_state_space']),
            discretize_state_space=str2bool(mdp_args['discretize_state_space']),
            reward=literal_eval(mdp_args['reward']),
            reward_rescale=float(mdp_args['reward_rescale']),
            target_velocity=literal_eval(mdp_args['target_velocity']),
            velocity_threshold=literal_eval(mdp_args['velocity_threshold']),
        )

        return mdp_params

    def parse_agent_params(self):
        """
            Parses agent parameters config file.
            Loads agent type (e.g. 'QL' or 'DQN') from
            train.config file and the respective parameters.

            Returns:
            -------
            * agent_type: str
                the type of the agent (e.g. 'QL' or 'DQN')

            * agent_params: ilurl.core.params object
                object containing the agent's parameters

        """
        # Load parameters form train.config.
        train_config = configparser.ConfigParser()
        train_config.read(str(self.config_path))

        # Read agent type: 'QL' or 'DQN'.
        agent_type = train_config['agent_type']['agent_type']

        if agent_type not in AGENT_TYPES:
                raise ValueError(f'''
                    Agent type must be in {AGENT_TYPES}.
                    Got {agent_type} type instead.''')

        if agent_type == 'QL':
            agent_params = self._parse_ql_params(train_config)
        elif agent_type == 'DQN':
            agent_params = self._parse_dqn_params(train_config)
        else:
            raise ValueError('Unkown agent type.')

        return agent_type, agent_params

    def _parse_ql_params(self, train_config):
        """
            Parses Q-learning parameters (ql_args) from config file located
            at self.config_path and returns a ilurl.core.params.QLParams
            object with the parsed parameters.
        """

        ql_args = train_config['ql_args']

        ql_params = QLParams(
                        lr_decay_power_coef=float(ql_args['lr_decay_power_coef']),
                        eps_decay_power_coef=float(ql_args['eps_decay_power_coef']),
                        gamma=float(ql_args['gamma']),
                        choice_type=ql_args['choice_type'],
                        replay_buffer=str2bool(ql_args['replay_buffer']),
                        replay_buffer_size=int(ql_args['replay_buffer_size']),
                        replay_buffer_batch_size=int(ql_args['replay_buffer_batch_size']),
                        replay_buffer_warm_up=int(ql_args['replay_buffer_warm_up']),
        )

        return ql_params

    def _parse_dqn_params(self, train_config):
        """
            Parses Deep Q-Network parameters (dqn_args) from config file located
            at self.config_path and returns a ilurl.core.params.DQNParams
            object with the parsed parameters.
        """

        dqn_args = train_config['dqn_args']

        dqn_params = DQNParams(
                        learning_rate= float(dqn_args['learning_rate']),
                        gamma=float(dqn_args['discount']),
                        batch_size=int(dqn_args['batch_size']),
                        prefetch_size=int(dqn_args['prefetch_size']),
                        target_update_period=int(dqn_args['target_update_period']),
                        samples_per_insert=float(dqn_args['samples_per_insert']),
                        min_replay_size=int(dqn_args['min_replay_size']),
                        max_replay_size=int(dqn_args['max_replay_size']),
                        importance_sampling_exponent=float(dqn_args['importance_sampling_exponent']),
                        priority_exponent=float(dqn_args['priority_exponent']),
                        n_step=int(dqn_args['n_step']),
                        epsilon=float(dqn_args['epsilon']),
        )

        return dqn_params


config_parser = Parser()
