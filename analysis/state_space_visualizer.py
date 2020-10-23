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
ILU_PATH = environ['ALTRL_HOME']
NUM_PHASES = 2  #TODO: RELAX THIS ASSUMPTION

# TODO: make this an input argument
# LOG_PATH = 'data/emissions/20200916183156.533079/grid_6_20200916-1831561600277516.627527/logs/train_log.json'
# LOG_PATH = 'data/emissions/20200828015100.697569/grid_20200828-0151021598575862.5034564/logs/train_log.json'
# LOG_PATH = "data/emissions/20200916161353.347374/grid_6_20200916-1613531600269233.4435577/logs/train_log.json"

# LOG_PATH = 'data/experiments/intbins/wtime/intersection_20201009-2004551602270295.8987248/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/ss/intersection_20201009-2344491602283489.5916183/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/sc/intersection_20201009-1125051602239105.9357407/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/ddel/intersection_20201012-1944331602528273.319772/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/del/intersection_20201008-2241231602193283.2883132/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/pss/intersection_20201008-2356301602197790.8946025/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/pss/intersection_20201021-1922561603304576.0938213/logs/train_log.json'
# LOG_PATH = 'data/experiments/intbins/flow/intersection_20201021-1919381603304378.5587554/logs/train_log.json'
LOG_PATH = 'data/experiments/intbins/queue/intersection_20201009-1552361602255156.0307872/logs/train_log.json'
if __name__ == '__main__':

    log_path = Path(ILU_PATH) / LOG_PATH
    config_path = log_path.parent.parent / 'config'
    quantile = (0.15, 0.5, 0.75, 0.85)
    # TODO: go to config and determine features
    train_config = configparser.ConfigParser()
    train_config.read((config_path / 'train.config').as_posix())

    # TODO: go to config and network
    network = train_config.get('train_args', 'network')
    features = eval(train_config.get('mdp_args', 'features'))

    # 1. Create log_path
    target_path = log_path.parent.parent / 'log_plots'
    target_path.mkdir(mode=0o777, exist_ok=True)

    # TODO: Aggregate multiple experiments
    with log_path.open(mode='r') as f:

        json_data = json.load(f)

        states = json_data['states']
        states = pd.DataFrame(states)

        # Interates per intersection
        for (tid, tdata) in states.iteritems():

            # assumptions there are always two phases
            labels = [f'{label}_{n}' for n in range(2) for label in features]
            df = pd.DataFrame(tdata.to_list(), columns=labels)
            # Num. plots == Num. features  x Num. intersection
            # Num. series == Num. labels in feature
            for feature in features:
                # Delay.
                fig = plt.figure()
                fig.set_size_inches(FIGURE_X, FIGURE_Y)

                for num in range(2):
                    label = f'{feature}_{num}'
                    sns.distplot(df[label], hist=False, kde=True, label=label, kde_kws = {'linewidth': 3})

                plt.xlabel(f'{snakefy(feature)}')
                plt.ylabel('Density')
                plt.title(f'{snakefy(network)}: Intersection {tid}, {snakefy(feature)} feature')
                plt.savefig((target_path / f'{tid}-{feature}.png').as_posix(),
                            bbox_inches='tight', pad_inches=0)
                plt.close()

        # Save quantile plots
        qdf = df.quantile(quantile)
        print(qdf.head(len(quantile)))
        qdf.to_csv((target_path / 'quantile.csv').as_posix(), sep='|', encoding='utf-8')
