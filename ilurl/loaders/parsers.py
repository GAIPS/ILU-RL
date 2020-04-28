from os import environ
from pathlib import Path

import configparser

from ilurl.core.params import QLParams

ILURL_PATH = Path(environ['ILURL_HOME'])
CONFIG_PATH = ILURL_PATH / 'config/agents.config'

def str2bool(v, exception=None):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def parse_ql_params():
    """
        Parses Q-learning parameters (ql_args) from config file located
        at 'CONFIG_PATH' and returns a ilurl.core.params.QLParams
        object with the parsed parameters.
    """

    # Load config file with parameters.
    agents_config = configparser.ConfigParser()
    agents_config.read(str(CONFIG_PATH))

    ql_args = agents_config['ql_args']

    ql_params = QLParams(
                    gamma=float(ql_args['gamma']),
                    choice_type=ql_args['choice_type'],
                    replay_buffer=str2bool(ql_args['replay_buffer']),
                    replay_buffer_size=int(ql_args['replay_buffer_size']),
                    replay_buffer_batch_size=int(ql_args['replay_buffer_batch_size']),
                    replay_buffer_warm_up=int(ql_args['replay_buffer_warm_up']),
    )

    return ql_params