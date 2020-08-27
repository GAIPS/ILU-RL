import json
import configparser
from os import environ
from pathlib import Path
from ast import literal_eval

from shutil import copyfile

from ilurl.agents.factory import AgentFactory
from ilurl.params import (QLParams,
                          DQNParams,
                          R2D2Params,
                          DDPGParams,
                          MDPParams,
                          TrainParams)

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
            self.config_path and returns a ilurl.params.TrainParams object
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
            at self.config_path and returns a ilurl.params.MDPParams
            object with the parsed parameters.
        """
        # Load config file with parameters.
        train_config = configparser.ConfigParser()
        train_config.read(str(self.config_path))

        mdp_args = train_config['mdp_args']

        time_period = int(mdp_args['time_period']) if not isNone(mdp_args['time_period']) else None

        mdp_params = MDPParams(
            discount_factor=float(mdp_args['discount_factor']),
            action_space=literal_eval(mdp_args['action_space']),
            features=literal_eval(mdp_args['features']),
            category_counts=json.loads(mdp_args['category_counts']),
            category_delays=json.loads(mdp_args['category_delays']),
            category_speeds=json.loads(mdp_args['category_speeds']),
            category_speed_scores=json.loads(mdp_args['category_speed_scores']),
            category_pressures=json.loads(mdp_args['category_pressures']),
            category_average_pressures=json.loads(mdp_args['category_average_pressures']),
            category_queues=json.loads(mdp_args['category_queues']),
            category_times=json.loads(mdp_args['category_times']),
            normalize_state_space=str2bool(mdp_args['normalize_state_space']),
            discretize_state_space=str2bool(mdp_args['discretize_state_space']),
            reward=literal_eval(mdp_args['reward']),
            reward_rescale=float(mdp_args['reward_rescale']),
            velocity_threshold=literal_eval(mdp_args['velocity_threshold']),
            time_period=time_period,
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

        # Check if agent exists.
        AgentFactory.get(agent_type)

        if agent_type == 'QL':
            agent_params = self._parse_ql_params(train_config)
        elif agent_type == 'DQN':
            agent_params = self._parse_dqn_params(train_config)
        elif agent_type == 'R2D2':
            agent_params = self._parse_r2d2_params(train_config)
        elif agent_type == 'DDPG':
            agent_params = self._parse_ddpg_params(train_config)
        else:
            raise ValueError('Unkown agent type.')

        return agent_type, agent_params

    def _parse_ql_params(self, train_config):
        """
            Parses Q-learning parameters (ql_args) from config file located
            at self.config_path and returns a ilurl.params.QLParams
            object with the parsed parameters.
        """

        ql_args = train_config['ql_args']

        ql_params = QLParams(
                        lr_decay_power_coef=float(ql_args['lr_decay_power_coef']),
                        eps_decay_power_coef=float(ql_args['eps_decay_power_coef']),
                        choice_type=ql_args['choice_type'],
                        replay_buffer=str2bool(ql_args['replay_buffer']),
                        replay_buffer_size=int(ql_args['replay_buffer_size']),
                        replay_buffer_batch_size=int(ql_args['replay_buffer_batch_size']),
                        replay_buffer_warm_up=int(ql_args['replay_buffer_warm_up']),
        )

        # print(ql_params)

        return ql_params

    def _parse_dqn_params(self, train_config):
        """
            Parses Deep Q-Network parameters (dqn_args) from config file located
            at self.config_path and returns a ilurl.params.DQNParams
            object with the parsed parameters.
        """

        dqn_args = train_config['dqn_args']

        dqn_params = DQNParams(
                        learning_rate=float(dqn_args['learning_rate']),
                        batch_size=int(dqn_args['batch_size']),
                        prefetch_size=int(dqn_args['prefetch_size']),
                        target_update_period=int(dqn_args['target_update_period']),
                        samples_per_insert=float(dqn_args['samples_per_insert']),
                        min_replay_size=int(dqn_args['min_replay_size']),
                        max_replay_size=int(dqn_args['max_replay_size']),
                        importance_sampling_exponent=float(dqn_args['importance_sampling_exponent']),
                        priority_exponent=float(dqn_args['priority_exponent']),
                        n_step=int(dqn_args['n_step']),
                        epsilon_init=float(dqn_args['epsilon_init']),
                        epsilon_final=float(dqn_args['epsilon_final']),
                        epsilon_schedule_timesteps=int(dqn_args['epsilon_schedule_timesteps']),
                        torso_layers=json.loads(dqn_args['torso_layers']),
                        head_layers=json.loads(dqn_args['head_layers']),
        )

        # print(dqn_params)

        return dqn_params

    def _parse_r2d2_params(self, train_config):
        """
            Parses R2D2 parameters (r2d2_args) from config file located
            at self.config_path and returns a ilurl.params.R2D2Params
            object with the parsed parameters.
        """

        r2d2_args = train_config['r2d2_args']

        r2d2_params = R2D2Params(
                        burn_in_length=int(r2d2_args['burn_in_length']),
                        trace_length=int(r2d2_args['trace_length']),
                        replay_period=int(r2d2_args['replay_period']),  
                        batch_size=int(r2d2_args['batch_size']), 
                        prefetch_size=int(r2d2_args['prefetch_size']),     
                        target_update_period=int(r2d2_args['target_update_period']),
                        importance_sampling_exponent=float(r2d2_args['importance_sampling_exponent']),
                        priority_exponent=float(r2d2_args['priority_exponent']),
                        learning_rate= float(r2d2_args['learning_rate']),
                        min_replay_size=int(r2d2_args['min_replay_size']),
                        max_replay_size=int(r2d2_args['max_replay_size']),
                        samples_per_insert=float(r2d2_args['samples_per_insert']),
                        store_lstm_state=str2bool(r2d2_args['store_lstm_state']),
                        max_priority_weight=float(r2d2_args['max_priority_weight']),
                        epsilon_init=float(r2d2_args['epsilon_init']),
                        epsilon_final=float(r2d2_args['epsilon_final']),
                        epsilon_schedule_timesteps=int(r2d2_args['epsilon_schedule_timesteps']),
                        rnn_hidden_size=int(r2d2_args['rnn_hidden_size']),
                        head_layers=json.loads(r2d2_args['head_layers']),
        )

        # print(r2d2_params)

        return r2d2_params

    def _parse_ddpg_params(self, train_config):
        """
            Parses DDPG parameters (ddpg_args) from config file located
            at self.config_path and returns a ilurl.params.DDPGParams
            object with the parsed parameters.
        """

        ddpg_args = train_config['ddpg_args']

        ddpg_params = DDPGParams(
                        batch_size=int(ddpg_args['batch_size']),
                        prefetch_size=int(ddpg_args['prefetch_size']),
                        target_update_period=int(ddpg_args['target_update_period']),
                        min_replay_size=int(ddpg_args['min_replay_size']),
                        max_replay_size=int(ddpg_args['max_replay_size']),
                        samples_per_insert=float(ddpg_args['samples_per_insert']),
                        n_step=int(ddpg_args['n_step']),
                        sigma_init=float(ddpg_args['sigma_init']),
                        sigma_final=float(ddpg_args['sigma_final']),
                        sigma_schedule_timesteps=int(ddpg_args['sigma_schedule_timesteps']),
                        clipping=str2bool(ddpg_args['clipping']),
                        policy_layers=json.loads(ddpg_args['policy_layers']),
                        critic_layers=json.loads(ddpg_args['critic_layers']),
        )

        # print(ddpg_params)

        return ddpg_params


config_parser = Parser()
