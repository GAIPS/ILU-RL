"""This script helps setting sensible discretization bins

    * Run an experiment such as DQN.
    * Change path to train_log_folder/
      (i.e data/emissions/20200916183156.533079/grid_6_20200916-1831561600277516.627527/logs/train_log.json)

    * Will create experiment_folder/log_plots.
    * Will save plots for histograms.
    * Will save quantiles for distributions."""
from os import environ
from pathlib import Path
import json
from collections import defaultdict


import configparser
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from ilurl.utils.aux import snakefy
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# CONSTANTS
FIGURE_X = 6.0
FIGURE_Y = 4.0
ILU_PATH = environ['ILURL_HOME']
NUM_PHASES = 2  #TODO: RELAX THIS ASSUMPTION

# TODO: make this an input argument
# LOG_PATH = 'data/emissions/20200916183156.533079/grid_6_20200916-1831561600277516.627527/logs/train_log.json'
# LOG_PATH = 'data/emissions/20200828015100.697569/grid_20200828-0151021598575862.5034564/logs/train_log.json'
LOG_PATH = "data/emissions/20200916161353.347374/grid_6_20200916-1613531600269233.4435577/logs/train_log.json"

if __name__ == '__main__':

    log_path = Path(ILU_PATH) / LOG_PATH
    config_path = log_path.parent.parent / 'config' 
    quantile = (0.2, 0.4, 0.6, 0.8, 0.9)
    # TODO: go to config and determine features
    train_config = configparser.ConfigParser()
    train_config.read((config_path / 'train.config').as_posix())

    # TODO: go to config and network
    network = train_config.get('train_args', 'network')
    features = eval(train_config.get('mdp_args', 'features'))

    # 1. Create log_path
    target_path = log_path.parent.parent / 'log_plots'
    target_path.mkdir(mode=0o777, exist_ok=True)
    data = defaultdict(list)
    with log_path.open(mode='r') as f:

        json_data = json.load(f)

        states = json_data['states']
        states = pd.DataFrame(states)

        # Interates per intersection
        for (tl_id, tl_data) in states.iteritems():

            # assumptions there are always two phases
            labels = [f'{label}_{n}' for n in range(2) for label in features]
            tl_data = pd.DataFrame(tl_data.to_list(), columns=labels)

            # Num. plots == Num. features  x Num. intersection
            # Num. series == Num. labels in feature
            for feature in features:
                # Delay.
                fig = plt.figure()
                fig.set_size_inches(FIGURE_X, FIGURE_Y)

                for num in range(2):
                    label = f'{feature}_{num}'
                    print(label)
                    sns.distplot(tl_data[label], hist=False, kde=True, label=label, kde_kws = {'linewidth': 3})

                    data[feature] = np.concatenate((data[feature], tl_data[label].values))

                
                plt.xlabel(f'(Normalized) {snakefy(feature)}')
                plt.ylabel('Density')
                plt.title(f'{snakefy(network)}: Intersection {tl_id}, {snakefy(feature)} feature')
                plt.savefig((target_path / f'{tl_id}-{feature}.png').as_posix(),
                            bbox_inches='tight', pad_inches=0)
                plt.close()

        # Save quantile plots
        df = pd.DataFrame.from_dict(data)
        df = df.quantile(quantile)
        print(df.head(len(quantile)))
        df.to_csv((target_path / 'quantile.csv').as_posix(), sep='|', encoding='utf-8')
