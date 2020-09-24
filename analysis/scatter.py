"""This script makes a scatter plot state vs state

    TODO:
    -----
    1) Segregate phases: 1 axis => two phases.
    4) Relax speed and count to any features.
    5) Consolidate tls signals
    2) Label category digits
    3) Reward vs state space observations
    

    USAGE:
    -----

    From directory
    > python analysis/scatter.py 20200923143127.894361/intersection_20200923-1431271600867887.8995893/

"""
# core packages
import re
from pathlib import Path
from collections import defaultdict
import json
from os import environ
from glob import glob
import argparse

# third-party libs
import configparser
import matplotlib.pyplot as plt

# project dependencies
from ilurl.utils.aux import TIMESTAMP

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
    train_log_path = experiment_path / 'logs' / 'train_log.json'
    config_path = experiment_path / 'config' / 'train.config'

    phases = defaultdict(list)
    with train_log_path.open('r') as f:
        output = json.load(f)
    observation_spaces = output['observation_spaces']

    train_config = configparser.ConfigParser()
    train_config.read(config_path.as_posix())
    # TODO: Use the features to get the categories
    labels = eval(train_config.get('mdp_args', 'features'))
    network_id = train_config.get('train_args', 'network')



    category_speeds = eval(train_config.get('mdp_args', 'category_speeds'))
    category_counts = eval(train_config.get('mdp_args', 'category_counts'))
    for observation_space in observation_spaces:
        # TODO: Relax constraints for multiple tls
        for intersection_space in observation_space.values():
            for num, phase_space in enumerate(intersection_space):
                phases[num].append(phase_space)

    _, ax = plt.subplots()
    for i, label in enumerate(labels):
        if i == 0:
            ax.set_xlabel(label)
        elif i == 1:
            ax.set_ylabel(label)


    ax.vlines(category_speeds, 0, 1,
              transform=ax.get_xaxis_transform(),
              colors='tab:gray')

    ax.hlines(category_counts, 0, 1,
              transform=ax.get_yaxis_transform(),
              colors='tab:gray',
              label='states')

    colors = ['tab:blue', 'tab:red']
    N = 0
    for i, phase in phases.items():
        x, y = zip(*phase)
        N += len(x)
        ax.scatter(x, y, c=colors[i], label=f'phase#{i}')

    expid = experiment_path.parts[-1]
    result = re.search(TIMESTAMP, expid)

    if result:
        timestamp = result.group(0,)
    else:
        timetamp = expid
    ax.legend()
    ax.grid(True)
    plt.title(f'{network_id}\n{timestamp}:\nobservation space (N={N})')
    plt.savefig(experiment_path / 'scatter.png')
    plt.show()
