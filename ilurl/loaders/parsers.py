import json
from os import environ
from pathlib import Path
from ast import literal_eval

import configparser

from ilurl.core.params import QLParams, DQNParams, MDPParams

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

def parse_agent_params():
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
    train_config.read(str(TRAIN_CONFIG_PATH))

    # Read agent type: 'QL' or 'DQN'.
    agent_type = train_config['agent_type']['agent_type']

    if agent_type not in AGENT_TYPES:
            raise ValueError(f'''
                Agent type must be in {AGENT_TYPES}.
                Got {agent_type} type instead.''')

    if agent_type == 'QL':
        agent_params = parse_ql_params(train_config)
    elif agent_type == 'DQN':
        agent_params = parse_dqn_params(train_config)
    else:
        raise ValueError('Unkown agent type.')

    return agent_type, agent_params

def parse_ql_params(train_config):
    """
        Parses Q-learning parameters (ql_args) from config file located
        at 'TRAIN_CONFIG_PATH' and returns a ilurl.core.params.QLParams
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

def parse_dqn_params(train_config):
    """
        Parses Deep Q-Network parameters (dqn_args) from config file located
        at 'TRAIN_CONFIG_PATH' and returns a ilurl.core.params.DQNParams
        object with the parsed parameters.
    """

    dqn_args = train_config['dqn_args']

    dqn_params = DQNParams(
                    lr= float(dqn_args['lr']),
                    gamma=float(dqn_args['gamma']),
                    buffer_size=int(dqn_args['buffer_size']),
                    batch_size=int(dqn_args['batch_size']),
                    exp_initial_p=float(dqn_args['exp_initial_p']),
                    exp_final_p=float(dqn_args['exp_final_p']),
                    exp_schedule_timesteps=int(dqn_args['exp_schedule_timesteps']),
                    learning_starts=int(dqn_args['learning_starts']),
                    target_net_update_interval=int(dqn_args['target_net_update_interval']),
    )

    return dqn_params 

def parse_mdp_params():
    """
        Parses MDP parameters (mdp_args) from config file located
        at 'TRAIN_CONFIG_PATH' and returns a ilurl.core.params.MDPParams
        object with the parsed parameters.
    """

    # Load config file with parameters.
    agents_config = configparser.ConfigParser()
    agents_config.read(str(TRAIN_CONFIG_PATH))

    mdp_args = agents_config['mdp_args']

    mdp_params = MDPParams(
                    states=literal_eval(mdp_args['states']),
                    category_counts=json.loads(mdp_args['category_counts']),
                    category_speeds=json.loads(mdp_args['category_speeds']),
                    normalize_state_space=str2bool(mdp_args['normalize_state_space']),
                    discretize_state_space=str2bool(mdp_args['discretize_state_space']),
                    reward=literal_eval(mdp_args['reward'])
    )

    return mdp_params 