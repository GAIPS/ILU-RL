# core packages
import re
from pathlib import Path
from importlib import import_module
from collections import defaultdict, namedtuple
import json
from os import environ
import argparse
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0


if __name__ == '__main__':

    """ experiment_path = Path('/home/ppsantos/Desktop/20201117145441.965866')
    rollouts_path = experiment_path / 'rollouts_test.json'

    with rollouts_path.open('r') as f:
        data = json.load(f)

    id = str(data['id'][0])
    # print(output)

    print(len(data['observation_spaces'][id]))
    print(data['observation_spaces'][id][0])

    phase_1_counts = []
    phase_2_counts = []

    for t in data['observation_spaces'][id][0]:

        phase_1_counts.append(t['247123161'][0][1])
        phase_2_counts.append(t['247123161'][1][1])
    
    print(phase_1_counts)
    print(phase_2_counts)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # window_size = min(len(df_actions)-1, 40)

    plt.plot(phase_1_counts, label='Phase 1') # .rolling(window=window_size).mean()
    plt.plot(phase_2_counts, label='Phase 2') # .rolling(window=window_size).mean()

    plt.xlabel('Cycle')
    plt.ylabel('Average number of vehicles')

    plt.legend()

    plt.savefig('counts_cyclical.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('counts_cyclical.png', bbox_inches='tight', pad_inches=0)

    plt.close() """

    experiment_path = Path('/home/ppsantos/Desktop/20201117145441.965866')
    rollouts_path = experiment_path / 'rollouts_test.json'

    with rollouts_path.open('r') as f:
        data = json.load(f)

    id = str(data['id'][0])
    # print(output)

    # print(len(data['observation_spaces'][id]))
    # print(data['observation_spaces'][id][0])

    phase_1_counts = np.zeros((90,len(data['observation_spaces'][id][0])))
    phase_2_counts = np.zeros((90,len(data['observation_spaces'][id][0])))

    for i in range(90):

        phase_1_c = []
        phase_2_c = []
        for t in data['observation_spaces'][id][i]:

            phase_1_c.append(t['247123161'][0][1])
            phase_2_c.append(t['247123161'][1][1])

        phase_1_counts[i,:] = phase_1_c
        phase_2_counts[i,:] = phase_2_c
    
    print(phase_1_counts)
    print(phase_2_counts)

    phase_1_counts = np.mean(phase_1_counts, axis=0)
    phase_2_counts = np.mean(phase_2_counts, axis=0)

    print(phase_1_counts)
    print(phase_2_counts)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    plt.plot(phase_1_counts, label='Phase 1') # .rolling(window=window_size).mean()
    plt.plot(phase_2_counts, label='Phase 2') # .rolling(window=window_size).mean()

    plt.xlabel('Cycle')
    plt.ylabel('Average number of vehicles')

    plt.legend()

    plt.savefig('counts_cyclical.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('counts_cyclical.png', bbox_inches='tight', pad_inches=0)

    plt.close()    
