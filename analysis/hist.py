"""This script makes a histogram from the info.json output from experiment

    Use this script  to determine the best categorical breakdowns

    USAGE:
    -----
    From root directory with files saved on root
    > python analysis/hist.py 20200929124206.116251/intersection_20200929-1242061601379726.138305/

    UPDATE:
    -------
    2019-12-11
        * update normpdf function
        * deprecate TrafficLightQLGridEnv in favor of TrafficQLEnv

    2020-02-20
        * swap filename for pattern matching uniting many files at once

    2020-09-28
        * change to argument for histogram
"""
# core packages
from collections import defaultdict
from pathlib import Path
import json
from os import environ
import argparse


# third-party libs
import configparser
import dill
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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

ROOT_DIR = os.environ['ALTRL_HOME']

if __name__ == '__main__':
    args = get_arguments()
    experiment_path = Path(args.experiment_dir)
    train_log_path = experiment_path / 'logs' / 'train_log.json'
    config_path = experiment_path / 'config' / 'train.config'
    target_path = experiment_path / 'log_plots'
    target_path.mkdir(mode=0o777, exist_ok=True)

    # Retrieves features
    train_config = configparser.ConfigParser()
    train_config.read(config_path.as_posix())
    labels = eval(train_config.get('mdp_args', 'features'))

    # this loop acumulates experiments
    states = defaultdict(list)

    # Retrieves output data
    with train_log_path.open('r') as f:
        output = json.load(f)
    observation_spaces = output['observation_spaces']
    rewards = output['rewards']

    # Agregages by feature
    for observation_space in observation_spaces:
        for phase_space in observation_space.values():
            for features in phase_space:
                for label, val in zip(labels, features):
                    states[label].append(val)

    # plot building
    num_bins = 100
    # percentile separators: low, medium and high
    percentile_separators = range(10, 100, 10)
    # perceptile_colors = ('yellow', 'green')
    for label, values in states.items():
        fig, ax = plt.subplots()

        # mean and standard deviation of the distribution
        mu = np.mean(values)
        sigma = np.std(values)
        # the histogram of the data
        # values_normalized = [
        #     round((v - mu) / sigma, 2) for v in values
        # ]
        # Define quantiles for the histogram
        # ignore lower and higher values
        quantiles = np.percentile(values, percentile_separators)
        print(f"#########{label}##########")
        print(f"min:\t{np.round(min(values), 2)}")
        for i, q in enumerate(quantiles):
            # color = perceptile_colors[i]
            color = 'tab:purple'
            p = percentile_separators[i]
            legend = f'p {int(p)} %'
            ax.axvline(x=float(q),
                        markerfacecoloralt=color,
                        label=legend)

            # Tweak spacing to prevent clipping of ylabel
            print(f"{p}%\t{np.round(q, 2)}")
        print(f"max:\t{np.round(max(values), 2)}")

        n, bins, patches = ax.hist(
            values,
            num_bins,
            density=mu,
            facecolor='blue',
            alpha=0.5
        )

        # add a 'best fit' line
        y = norm.pdf(bins, mu, sigma)
        ax.plot(bins, y, 'r--')
        ax.set_xlabel(label)
        ax.set_ylabel('Probability')
        title = f"Histogram of {label}"
        title = f"{title}\n$\mu$={round(mu, 2)},"
        title = f"{title}$\sigma$={round(sigma,2)}"
        ax.set_title(title)

        plt.subplots_adjust(left=0.15)
        plt.savefig(target_path / f'{label}.png')
        plt.show()

        # Tweak spacing to prevent clipping of ylabel
        # plt.subplots_adjust(left=0.15)
        # print(f"#########{label}##########")
        # print(f"min:\t{np.round(quantiles[0] * sigma + mu, 2)}")
        # print(f"{percentile_separators[1]}%\t{np.round(quantiles[1] * sigma + mu, 2)}")
        # print(f"{percentile_separators[2]}%\t{np.round(quantiles[2] * sigma + mu, 2)}")
        # print(f"max:\t{np.round(quantiles[-1] * sigma + mu, 2)}")

        # plt.show()
        # plt.savefig(target_path / f'{label}.png')
