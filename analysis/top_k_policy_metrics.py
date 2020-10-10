import os
import json
import pandas as pd
import argparse
import tarfile
import numpy as np
from pathlib import Path
from scipy import stats
import configparser


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script reports the metrics for the top-K policies.
        """
    )
    parser.add_argument('experiment_root_folder', type=str, nargs='?',
                        help='Experiment root folder.')
    parser.add_argument('--num_samples', type=int, required=False,
                    help='K value.', default=3)

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/top_k_policy_metrics.py):')
    print('\tExperiment root folder: {0}\n'.format(args.experiment_root_folder))
    print('\tK: {0}\n'.format(args.num_samples))

def main():

    print('\nRUNNING analysis/top_k_policy_metrics.py\n')

    args = get_arguments()
    print_arguments(args)

    exp_name = os.path.basename(args.experiment_root_folder)
    exp_name = exp_name.split('.')[0] + '.' + exp_name.split('.')[1]

    df_metrics = pd.read_csv(tarfile.open(args.experiment_root_folder).extractfile(
                            f'{exp_name}/plots/test/{exp_name}_metrics.csv'))

    df_metrics_grouped = df_metrics.groupby(['train_run']).mean()

    top_k = df_metrics_grouped.nsmallest(args.num_samples, 'travel_time')
    top_k_mean = top_k.mean()

    print(f'Top-{args.num_samples} policies (w.r.t. travel_time):')
    print('-'*30)
    print(f'Speed: {top_k_mean["speed"]}')
    print(f'Waiting time: {top_k_mean["waiting_time"]}')
    print(f'Travel time: {top_k_mean["travel_time"]}')

if __name__ == "__main__":
    main()
