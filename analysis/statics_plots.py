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

# LOW.
LOW = [
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228224017.932660.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228230214.692122.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228232626.787120.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201229000005.632899.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201229005341.245560.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201229012947.338313.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201229015557.365404.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201229023313.083746.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201229025618.406532.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201229033416.979982.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201229120831.284453.tar.gz',
]

# HIGH.
HIGH = [
    '/home/psantos/ILU/ILU-RL-2/data/emissions/20201228001604.039929.tar.gz',
    '/home/psantos/ILU/ILU-RL-2/data/emissions/20201228010932.438989.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228130416.380865.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228140236.438750.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228145547.916634.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228161354.315079.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228170044.551348.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228174046.508160.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228184009.906798.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228191858.022150.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228204856.111856.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201228220214.960041.tar.gz',
]

""" # BIASED.
BIASED = [
    '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_biased/20201007002101.781567.tar.gz',
    '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_biased/20201007005757.268721.tar.gz',
    '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_biased/20201011190750.886942.tar.gz',
    '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_biased/20201012172252.698076.tar.gz',
    '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_biased/20201012173647.918092.tar.gz',
    '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_biased/20201012180351.517013.tar.gz',
    '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_biased/20201012182210.354383.tar.gz',
] """

Xs_h = [12,14,15,16,17,18,19,20,22,24,26,28] 
Xs_l = [12,14,15,16,17,18,20,22,24,26,28]


def calculate_CI_bootstrap(x_hat, samples, num_resamples=20000):
    """
        Calculates 95 % interval using bootstrap.

        REF: https://ocw.mit.edu/courses/mathematics/
            18-05-introduction-to-probability-and-statistics-spring-2014/
            readings/MIT18_05S14_Reading24.pdf

    """
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    means = np.mean(resampled, axis=0)
    diffs = means - x_hat
    bounds = [x_hat - np.percentile(diffs, 5), x_hat - np.percentile(diffs, 95)]

    return bounds

def main():

    print('\nRUNNING analysis/statics_plots.py\n')

    # Prepare output folder.
    os.makedirs('analysis/plots/statics/', exist_ok=True)

    # Open dataframes (low demand).
    low_travel_times = []
    low_travel_times_bounds = []
    low_waiting_times = []
    low_waiting_times_bounds = []
    for exp_path in LOW:

        exp_name = Path(exp_path).name.split('.')[0] + \
                    '.' + Path(exp_path).name.split('.')[1]

        tar = tarfile.open(exp_path)
        tar_file = tar.extractfile("{0}/plots/test/{1}_metrics.csv".format(exp_name, exp_name))

        df = pd.read_csv(tar_file, header=[0, 1], index_col=0)

        df_mean = df.describe()

        mean_travel_time = df_mean['travel_time'].iloc[1,0]
        low_travel_times.append(mean_travel_time)
        travel_times = df['travel_time'].to_numpy().flatten()
        low_travel_times_bounds.append(calculate_CI_bootstrap(mean_travel_time, travel_times))

        mean_waiting_time = df_mean['waiting_time'].iloc[1,0]
        low_waiting_times.append(mean_waiting_time)
        waiting_times = df['waiting_time'].to_numpy().flatten()
        low_waiting_times_bounds.append(calculate_CI_bootstrap(mean_waiting_time, waiting_times))

    # Open dataframes (high demand).
    high_travel_times = []
    high_travel_times_bounds = []
    high_waiting_times = []
    high_waiting_times_bounds = []
    for exp_path in HIGH:

        exp_name = Path(exp_path).name.split('.')[0] + \
                    '.' + Path(exp_path).name.split('.')[1]

        tar = tarfile.open(exp_path)
        tar_file = tar.extractfile("{0}/plots/test/{1}_metrics.csv".format(exp_name, exp_name))

        df = pd.read_csv(tar_file, header=[0, 1], index_col=0)

        df_mean = df.describe()

        mean_travel_time = df_mean['travel_time'].iloc[1,0]
        high_travel_times.append(mean_travel_time)
        travel_times = df['travel_time'].to_numpy().flatten()
        high_travel_times_bounds.append(calculate_CI_bootstrap(mean_travel_time, travel_times))

        mean_waiting_time = df_mean['waiting_time'].iloc[1,0]
        high_waiting_times.append(mean_waiting_time)
        waiting_times = df['waiting_time'].to_numpy().flatten()
        high_waiting_times_bounds.append(calculate_CI_bootstrap(mean_waiting_time, waiting_times))

    # print(low_travel_times)
    # print(low_travel_times_bounds)
    # print(low_waiting_times)
    # print(low_waiting_times_bounds)


    # print(dfs_h)

    # Travel time.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    low_travel_times_bounds = np.array(low_travel_times_bounds).T
    low_travel_times_bounds = np.flip(low_travel_times_bounds, axis=0)
    error_lengths = np.abs(np.subtract(low_travel_times_bounds, low_travel_times))
    plt.errorbar(Xs_l, low_travel_times, yerr=error_lengths, capsize=3, label='Low')

    high_travel_times_bounds = np.array(high_travel_times_bounds).T
    high_travel_times_bounds = np.flip(high_travel_times_bounds, axis=0)
    error_lengths = np.abs(np.subtract(high_travel_times_bounds, high_travel_times))
    plt.errorbar(Xs_h, high_travel_times, yerr=error_lengths, capsize=3, label='High')

    plt.xlabel('Phase 1 allocation')
    plt.legend()
    plt.ylabel('Travel time (s)')
    
    plt.savefig('analysis/plots/statics/travel_time.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/statics/travel_time.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()

    # Waiting time.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    low_waiting_times_bounds = np.array(low_waiting_times_bounds).T
    low_waiting_times_bounds = np.flip(low_waiting_times_bounds, axis=0)
    error_lengths = np.abs(np.subtract(low_waiting_times_bounds, low_waiting_times))
    plt.errorbar(Xs_l, low_waiting_times, yerr=error_lengths, capsize=3, label='Low')

    high_waiting_times_bounds = np.array(high_waiting_times_bounds).T
    high_waiting_times_bounds = np.flip(high_waiting_times_bounds, axis=0)
    error_lengths = np.abs(np.subtract(high_waiting_times_bounds, high_waiting_times))
    plt.errorbar(Xs_h, high_waiting_times, yerr=error_lengths, capsize=3, label='High')

    plt.xlabel('Phase 1 allocation')
    plt.legend()
    plt.ylabel('Waiting time (s)')
    
    plt.savefig('analysis/plots/statics/waiting_time.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/statics/waiting_time.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()

if __name__ == "__main__":
    main()