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

DDPG_MIN_DELAY = [59.689, 59.755, 59.995, 60.149, 60.485, 60.812, 61.243]

Xs = [1,2,4,6,8,10,12]

GRAY_COLOR = (0.37,0.37,0.37)
GRAY_COLOR_2 = (0.43,0.43,0.43)

def main():

    # Travel time.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    plt.plot(Xs, DDPG_MIN_DELAY, label='DDPG + Min. delay')

    # Webster.
    plt.axhline(y=61.477, color=GRAY_COLOR, linestyle='--', label='Webster (top-12)')

    # Max-pressure.
    # plt.axhline(y=55.569, color='r', linestyle='--', label='Max-pressure')

    # Actuated.
    plt.axhline(y=58.545, color=GRAY_COLOR, linestyle='dotted', label='Actuated (top-12)')

    plt.xticks(ticks=[1,2,4,6,8,10,12], labels=['k=1', 'k=2', 'k=4', 'k=6', 'k=8', 'k=10', 'k=12'])

    #plt.ylim(59.5, 61.75)

    plt.xlabel('Top-k policies sets')
    plt.legend(loc=4)
    plt.ylabel('Average travel time (s)')
    
    plt.savefig('analysis/plots/top_metrics_plot.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('analysis/plots/top_metrics_plot.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()

if __name__ == "__main__":
    main()