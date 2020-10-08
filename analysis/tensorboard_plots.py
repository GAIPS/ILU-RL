import os
import re
import random
import tarfile
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
import configparser

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script creates training losses plots, similar do what is displayed in tensorboard.
        """
    )
    parser.add_argument('--experiment_root_folder', help='Experiment root folder.', required=True)
    parser.add_argument('--num_samples', type=int, required=False,
                    help='Number of train runs to sample and plot.', default=5)

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/tensorboard_plots.py):')
    print('\tExperiment root folder: {0}\n'.format(args.experiment_root_folder))
    print('\tNumber of samples: {0}\n'.format(args.num_samples))

def main():

    print('\nRUNNING analysis/tensorboard_plots.py\n')

    args = get_arguments()
    print_arguments(args)

    if Path(args.experiment_root_folder).suffix == '.gz':
        raise ValueError('Please uncompress folder first.')

    experiment_names = list(p for p in Path(args.experiment_root_folder).rglob('*-learning.csv'))

    if len(experiment_names) < args.num_samples:
        raise ValueError('num_samples argument should be <= than the number of training runs.')

    # Create output directory.
    OUTPUTS_FOLDER = args.experiment_root_folder + '/tensorboard_plots/'
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    # Get agent_type from config file.
    config_path = list(c for c in Path(args.experiment_root_folder).rglob('train.config'))[0]
    train_config = configparser.ConfigParser()
    train_config.read(config_path)
    agent_type = train_config['agent_type']['agent_type']

    # Randomly sample train runs.
    experiment_names = random.sample(experiment_names, k=args.num_samples)

    # Get columns names.
    cols = pd.read_csv(experiment_names[0]).columns

    def get_float_from_tensor(teststring):
        return float(re.findall("\d+\.\d+", teststring)[0]) # hack.

    for col in cols:

        if col in ('steps', 'walltime', 'step'):
            continue

        print(f'Creating plot for column "{col}":')
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for idx, exp in enumerate(experiment_names):
            print(f'\tProcessing {exp}')

            # Read data.
            df = pd.read_csv(exp)

            if agent_type in ('QL',): 
                # Non-tensor data: QL data is already in pythonic
                # format so nothing needs to be done here.
                pass
            else:
                # Values are tensor as strings so we need to convert them to floats.
                df[col] = df[col].apply(get_float_from_tensor)

            plt.plot(df[col], label=f'Train sample {idx}')

        # plt.xlim(0,500)
        plt.legend()
        plt.xlabel('Learning cycle')
        plt.ylabel(f'{col}')
        plt.savefig(OUTPUTS_FOLDER + f"{col}.png", bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUTS_FOLDER + f"{col}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    main()
