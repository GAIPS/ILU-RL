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


RANDOM_PATH = '/home/ppsantos/ILU/ILU-RL/data/experiments/random.csv'
DELAY_PATH = '/home/ppsantos/ILU/ILU-RL/data/experiments/delay.csv'
SPEED_COUNT_PATH = '/home/ppsantos/ILU/ILU-RL/data/experiments/speed_count.csv'
SPEED_SCORE_PATH = '/home/ppsantos/ILU/ILU-RL/data/experiments/speed_score.csv'
WAITING_TIME_PATH = '/home/ppsantos/ILU/ILU-RL/data/experiments/waiting_time.csv'


FIGURE_X = 6.0
FIGURE_Y = 4.0

if __name__ == '__main__':

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

    df_delay = pd.read_csv(DELAY_PATH)
    df_delay = df_delay.groupby('train_run').mean()

    df_wait = pd.read_csv(WAITING_TIME_PATH)
    df_wait = df_wait.groupby('train_run').mean()

    df_random = pd.read_csv(RANDOM_PATH)

    df_speed_count = pd.read_csv(SPEED_COUNT_PATH)
    df_speed_count = df_speed_count.groupby('train_run').mean()

    df_speed_score = pd.read_csv(SPEED_SCORE_PATH)
    df_speed_score = df_speed_score.groupby('train_run').mean()

    """
        Travel time.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_delay['travel_time'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Delay')
    sns.distplot(df_wait['travel_time'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Waiting time')
    sns.distplot(df_random['travel_time'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Random')
    sns.distplot(df_speed_count['travel_time'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Speed count')
    sns.distplot(df_speed_score['travel_time'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Speed score')

    plt.legend()
    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.title('Travel time (DDPG agent, intersection, variable)')
    plt.savefig('test_stat_tt.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Speed.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_delay['speed'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Delay')
    sns.distplot(df_wait['speed'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Waiting time')
    sns.distplot(df_random['speed'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Random')
    sns.distplot(df_speed_count['speed'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Speed count')
    sns.distplot(df_speed_score['speed'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Speed score')

    plt.legend()
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Density')
    plt.title('Speed (DDPG agent, intersection, variable)')
    plt.savefig('test_stat_s.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Waiting time.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_delay['delay'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Delay')
    sns.distplot(df_wait['delay'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Waiting time')
    sns.distplot(df_random['delay'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Random')
    sns.distplot(df_speed_count['delay'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Speed count')
    sns.distplot(df_speed_score['delay'], hist=True, kde=True, norm_hist=True,
                 kde_kws = {'linewidth': 3}, label='Speed score')

    plt.legend()
    plt.xlabel('Waiting time (s)')
    plt.ylabel('Density')
    plt.title('Waiting time (DDPG agent, intersection, variable)')
    plt.savefig('test_stat_w.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Shapiro.
    shapiro_test = stats.shapiro(df_delay['travel_time'])
    print('delay shapiro:', shapiro_test)

    shapiro_test = stats.shapiro(df_wait['travel_time'])
    print('wait shapiro:', shapiro_test)

    shapiro_test = stats.shapiro(df_random['travel_time'])
    print('random shapiro:', shapiro_test)

    shapiro_test = stats.shapiro(df_speed_count['travel_time'])
    print('speed_count shapiro:', shapiro_test)

    shapiro_test = stats.shapiro(df_speed_score['travel_time'])
    print('speed_score shapiro:', shapiro_test)

    # Levene.
    print('Levene test:', stats.levene(df_delay['travel_time'], df_wait['travel_time'],
                                       df_random['travel_time'], df_speed_score['travel_time'],
                                       df_speed_count['travel_time']))

    # ANOVA + Tukey-HSD.
    print('ANOVA:', stats.f_oneway(df_delay['travel_time'], df_wait['travel_time'],
                                    df_speed_score['travel_time'], df_speed_count['travel_time']))

    print('ANOVA:', stats.f_oneway(df_delay['travel_time'], df_wait['travel_time'],
                                    df_speed_score['travel_time'], df_speed_count['travel_time'],
                                    df_random['travel_time']))

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    data = df_delay['travel_time'].tolist() + df_wait['travel_time'].tolist() + \
            df_speed_score['travel_time'].tolist() + df_speed_count['travel_time'].tolist() + \
            df_random['travel_time'].tolist()
    groups = ['Delay' for _ in range(len(df_delay['travel_time']))] + \
        ['WaitTime' for _ in range(len(df_wait['travel_time']))] + \
        ['SpeedScore' for _ in range(len(df_speed_score['travel_time']))] + \
        ['SpeedCount' for _ in range(len(df_speed_count['travel_time']))] + \
        ['Random' for _ in range(len(df_random['travel_time']))]

    print('TukeyHSD:', pairwise_tukeyhsd(data, groups))
