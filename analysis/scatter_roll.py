"""This script makes a scatter plot 

    * Phase vs phase for a given feature.
    * Feature vs feature for a given state.
    * Handles multiple intersections.
    * Reward vs state for single feature states.
    * Reward vs state for multiple feature states.

    USAGE:
    -----
    From directory
    > python analysis/scatter.py 20200923143127.894361/intersection_20200923-1431271600867887.8995893/

"""
# core packages
import re
from pathlib import Path
from importlib import import_module
from collections import defaultdict, namedtuple
import json
from os import environ
import argparse

# third-party libs
import configparser
import matplotlib.pyplot as plt

# project dependencies
from ilurl.utils.aux import TIMESTAMP, snakefy
from ilurl.utils.plots import (scatter_states, scatter_phases)
import ilurl.rewards as rew

MockMDP = namedtuple('MockMDP', 'reward reward_rescale')
ILURL_PATH = Path(environ['ILURL_HOME'])

def get_categories(network_id, labels):
    """Gets features' categories"""
    category_path = ILURL_PATH / 'data' / 'networks' / network_id / 'categories.json'
    categories = {}
    # Network/Categories
    with category_path.open('r') as f:
        categories = json.load(f)
    # Filter
    categories = {tls: {label: categ[label] for label in labels}
                    for tls, categ in categories.items()}
    return categories

def get_series(observation_space, labels, xylabels=[], xyphases=[]):
    """Gets series"""
    if len(xylabels) == 0:
        xylabels = list(range(len(labels)))
    else:
        if not set(xylabels).issubset(range(len(labels))):
            raise ValueError('Must xylabels must be in 0, 1, ..., len(labels)')
        else:
            xylabels = sorted(xylabels)

    if len(xyphases) == 0:
        if not set(xyphases).issubset({0, 1}):
            raise ValueError('Must xyphases must be in {0,1}')
    else:
        xyphases = range(2)

    if len(xylabels) == 1:
        xys = [(x, x) for x in xylabels]
    else:
        # Make all combinations 2x2
        xys = [(x, y)
               for x in xylabels
               for y in xylabels if y > x]

    ret = {} # Phase 1
    for x, y in xys:
        # Cross labels.
        x_labels = (labels[x], labels[y])
        ret[x_labels] = {}
        for observation_space in observation_spaces:
            for tl, intersection_space in observation_space.items():
                if tl not in ret[x_labels]:
                    ret[x_labels][tl] = defaultdict(list)

                for num, phase in enumerate(intersection_space):
                    # feature vs feature e.g Speed vs Count
                    point = [feat for i, feat in enumerate(phase) if i in (x, y)]
                    if x != y:
                        ret[x_labels][tl][num].append(point)
                    else:
                        ret[x_labels][tl][num] += point

    return ret

def get_reward_function(train_config):
    # 1) Get build rewards
    rewards_module = import_module('.rewards', package='ilurl')
    build_rewards = getattr(rewards_module, 'build_rewards')

    # 2) Parse train config
    reward = eval(train_config.get('mdp_args', 'reward'))
    reward_rescale = eval(train_config.get('mdp_args', 'reward_rescale'))
    mock_params = MockMDP(reward=reward, reward_rescale=reward_rescale)

    return build_rewards(mock_params)

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
        This script plots the state space (variables) w.r.t the phases
        seem by each agent.
        """
    )
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        help='Directory to the experiment')

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

if __name__ == '__main__':
    # this loop acumulates experiments
    args = get_arguments()
    experiment_path = Path(args.experiment_dir)
    rollouts_path = experiment_path / 'rollouts_test.json'
    config_path = experiment_path / 'intersection_20201020-1200311603191631.7173903/config' / 'train.config'
    target_path = experiment_path / 'log_plots'
    target_path.mkdir(mode=0o777, exist_ok=True)

    phases = defaultdict(list)
    with rollouts_path.open('r') as f:
        output = json.load(f)

    id = str(output['id'][0])
    # print(output)
    observation_spaces = output['observation_spaces'][id][0] + output['observation_spaces'][id][1] + output['observation_spaces'][id][2]
    print(len(observation_spaces))
    rewards = output['rewards'][id][0]

    train_config = configparser.ConfigParser()
    train_config.read(config_path.as_posix())
    # TODO: Use the features to get the categories
    network_id = train_config.get('train_args', 'network')
    labels = eval(train_config.get('mdp_args', 'features'))

    reward_function = get_reward_function(train_config)
    reward_is_penalty = '_min_' in train_config.get('mdp_args', 'reward')

    # categories = get_categories(network_id, labels)

    # Scatter from phase 0 vs phase 1
    for j, label in enumerate(labels):
        feature_series = get_series(observation_spaces, labels, xylabels=[j])
        _rewards = rewards if len(labels) == 1 else []
        scatter_phases(feature_series, label, # categories,
                       network=network_id, save_path=target_path,
                       rewards=_rewards, reward_is_penalty=reward_is_penalty)

    if len(labels) > 1:
        states_series = get_series(observation_spaces, labels)
        scatter_states(states_series, # categories,
                       network=network_id, save_path=target_path,
                       rewards=rewards, reward_is_penalty=reward_is_penalty,
                       reward_function=reward_function)

