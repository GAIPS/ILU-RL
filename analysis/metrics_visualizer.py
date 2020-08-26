import os
import json
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from scipy import stats
import statsmodels


CSV_PATH = '/home/pedro/ILU/ILU-RL/data/experiments/inter_train_run_variability/mean_metrics_per_eval.csv'
CSV_PATH_MAX_PRESSURE = '/home/pedro/ILU/ILU-RL/data/experiments/20200816220549.714847_grid_6.csv'
CSV_PATH_WEBSTER = '/home/pedro/ILU/ILU-RL/data/experiments/20200825122649.153399_grid_6.csv'


FIGURE_X = 6.0
FIGURE_Y = 4.0

if __name__ == '__main__':

    data = pd.read_csv(CSV_PATH)

    """ fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(data['travel_time'][0:6], hist=True, kde=False, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Train run 1')
    sns.distplot(data['travel_time'][6:12], hist=True, kde=False, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Train run 2')
    sns.distplot(data['travel_time'][12:18], hist=True, kde=False, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Train run 3')
    sns.distplot(data['travel_time'][18:24], hist=True, kde=False, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Train run 4')
    sns.distplot(data['travel_time'][24:], hist=True, kde=False, norm_hist=True,
                kde_kws = {'linewidth': 3}, label='Train run 5')

    plt.legend()
    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.title('Travel time\n(per train run)')
    plt.savefig('variability_travel_time_metric_per_train_run.png', bbox_inches='tight', pad_inches=0)
    plt.close()



    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(data['travel_time'], hist=False, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3, 'color': 'black'})

    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.title('Travel time\n(All)')
    plt.savefig('variability_travel_time_metric.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    shapiro_test = stats.shapiro(data['travel_time'])
    print(shapiro_test)

    # Inter variability.
    global_mean = data['travel_time'].mean()

    eval_avgs = []
    eval_avgs.append(np.mean(data['travel_time'][0:6]))
    eval_avgs.append(np.mean(data['travel_time'][6:12]))
    eval_avgs.append(np.mean(data['travel_time'][12:18]))
    eval_avgs.append(np.mean(data['travel_time'][18:24]))
    eval_avgs.append(np.mean(data['travel_time'][24:]))
    eval_avgs = np.array(eval_avgs)

    inter_var = np.mean(np.abs(eval_avgs - global_mean))
    print('inter_var:', inter_var)

    # Intra variability.
    intra_avgs = []
    intra_avgs.append(np.mean(np.abs(data['travel_time'][0:6] - np.mean(data['travel_time'][0:6]))))
    intra_avgs.append(np.mean(np.abs(data['travel_time'][6:12] - np.mean(data['travel_time'][6:12]))))
    intra_avgs.append(np.mean(np.abs(data['travel_time'][12:18] - np.mean(data['travel_time'][12:18]))))
    intra_avgs.append(np.mean(np.abs(data['travel_time'][18:24] - np.mean(data['travel_time'][18:24]))))
    intra_avgs.append(np.mean(np.abs(data['travel_time'][24:] - np.mean(data['travel_time'][24:]))))

    intra_var = np.mean(intra_avgs)
    print('intra_avgs:', intra_var) """

    data_max_p = pd.read_csv(CSV_PATH_MAX_PRESSURE)
    data_web = pd.read_csv(CSV_PATH_WEBSTER)

    print('max_p', data_max_p['travel_time'].mean())
    print('web', data_web['travel_time'].mean())

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(data_max_p['travel_time'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Max-Pressure')
    sns.distplot(data_web['travel_time'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Webster')

    plt.legend()
    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.title('Travel time')
    plt.savefig('tset.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Shapiro.
    shapiro_test = stats.shapiro(data_max_p['travel_time'])
    print('max_p shapiro:', shapiro_test)

    shapiro_test = stats.shapiro(data_web['travel_time'])
    print('web shapiro:', shapiro_test)

    # Levene.
    print('Levene test:', stats.levene(data_max_p['travel_time'], data_web['travel_time']))

    # ANOVA + Tukey-HSD.
    print('ANOVA:', stats.f_oneway(data_max_p['travel_time'], data_web['travel_time']))

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    data = data_max_p['travel_time'].tolist() + data_web['travel_time'].tolist()
    groups = ['M' for _ in range(len(data_max_p['travel_time']))] + ['W' for _ in range(len(data_web['travel_time']))]
    print('TukeyHSD:', pairwise_tukeyhsd(data, groups))
