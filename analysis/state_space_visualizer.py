import os
import json
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

FIGURE_X = 6.0
FIGURE_Y = 4.0


PATH = '/home/pedro/ILU/ILU-RL/data/emissions/grid_6_20200710-1137061594377426.1875682/logs/train_log.json'


if __name__ == '__main__':

    os.makedirs('analysis/plots/', exist_ok=True)

    with open(PATH) as f:

        json_data = json.load(f)

        states = json_data['states']
        states = pd.DataFrame(states)

        for (columnName, columnData) in states.iteritems():

            columnData = pd.DataFrame(columnData.to_list(), columns=['s_0', 'c_0', 's_1','c_1'])

            # Speeds.
            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)

            sns.distplot(columnData['s_0'], hist=False, kde=True, label='phase 0',
                        kde_kws = {'linewidth': 3})

            sns.distplot(columnData['s_1'], hist=False, kde=True, label='phase 1',
                        kde_kws = {'linewidth': 3})

            plt.xlabel('(Normalized) Speed')
            plt.ylabel('Density')
            plt.title(f'grid_6: Intersection {columnName}, Speed feature')
            plt.savefig('analysis/plots/{0}-{1}.png'.format(columnName, 'speed'),
                        bbox_inches='tight', pad_inches=0)
            plt.close()

            # Counts.
            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)

            sns.distplot(columnData['c_0'], hist=False, kde=True, label='phase 0',
                        kde_kws = {'linewidth': 3})

            sns.distplot(columnData['c_1'], hist=False, kde=True, label='phase 1',
                        kde_kws = {'linewidth': 3})

            plt.xlabel('Counts')
            plt.ylabel('Density')
            plt.title(f'grid_6: Intersection {columnName}, Count feature')
            plt.savefig('analysis/plots/{0}-{1}.png'.format(columnName, 'count'),
                        bbox_inches='tight', pad_inches=0)
            plt.close()