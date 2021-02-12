import os
import tarfile
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import configparser
import tempfile
import shutil

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from scipy import stats
import statsmodels
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# Reward: point value vs variation.
""" EXPS_1 = [
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201019121818.176705.tar.gz',
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201125193144.350360.tar.gz',
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201111095306.078216.tar.gz',
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201126125852.876314.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201120162302.197023.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201128014914.794302.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201116213013.033741.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201128210244.601379.tar.gz',
]
LABEL_1 = '1'
EXPS_2 = [
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201212023738.171221.tar.gz', 
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201020043124.208366.tar.gz', 
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201126074429.705712.tar.gz', 
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201112231009.228489.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201127204521.381258.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201120211130.274071.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201227160831.998469.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201117022052.704456.tar.gz', 
]
LABEL_2 = '2' """

EXPS_1 = [
    # '/home/gvarela/ilu/ilurl/data/emissions/20201119150051.524184.tar.gz',
    # '/home/gvarela/ilu/ilurl/data/emissions/20201120015727.261873.tar.gz',
    # '/home/gvarela/ilu/ilurl/data/emissions/20201122163308.836162.tar.gz',
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201228005357.773218.tar.gz',
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201211170525.971782.tar.gz',
    # '/home/gvarela/ilu/ilurl/data/emissions/20201120091613.848725.tar.gz',
    # '/home/psantos/ILU/ILU-RL-2/data/emissions/20201228122903.543258.tar.gz',

    '/home/psantos/ILU/ILU-RL-2/data/emissions/20201228185736.808873.tar.gz',
    '/home/gvarela/ilu/ilurl/data/emissions/20201123205612.190239.tar.gz',
    '/home/gvarela/ilu/ilurl/data/emissions/20201125093216.415102.tar.gz',
    '/home/gvarela/ilu/ilurl/data/emissions/20201124091359.128610.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201212094406.166684.tar.gz',
    '/home/gvarela/ilu/ilurl/data/emissions/20201124035929.239557.tar.gz',
    '/home/gvarela/ilu/ilurl/data/emissions/20201124203350.027420.tar.gz',
]
LABEL_1 = 'QL'
EXPS_2 = [
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201019201927.705392.tar.gz', 
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201019121818.176705.tar.gz', 
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201212023738.171221.tar.gz', 
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201020120029.196691.tar.gz',
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201125193144.350360.tar.gz', 
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201020043124.208366.tar.gz', 
    # '/home/psantos/ILU/ILU-RL/data/emissions/20201125120732.210657.tar.gz', 

    '/home/psantos/ILU/ILU-RL/data/emissions/20201120112049.546333.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201120162302.197023.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201127204521.381258.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201121022041.160455.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201128014914.794302.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201120211130.274071.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201127164149.877018.tar.gz', 
]
LABEL_2 = 'DQN'
EXPS_3 = [
    #'/home/psantos/ILU/ILU-RL/data/emissions/20201112104325.247084.tar.gz', 
    #'/home/psantos/ILU/ILU-RL/data/emissions/20201111095306.078216.tar.gz', 
    #'/home/psantos/ILU/ILU-RL/data/emissions/20201126074429.705712.tar.gz', 
    #'/home/psantos/ILU/ILU-RL/data/emissions/20201113074152.431465.tar.gz',
    #'/home/psantos/ILU/ILU-RL/data/emissions/20201126125852.876314.tar.gz', 
    #'/home/psantos/ILU/ILU-RL/data/emissions/20201112231009.228489.tar.gz', 
    #'/home/psantos/ILU/ILU-RL/data/emissions/20201126023707.247087.tar.gz', 

    '/home/psantos/ILU/ILU-RL/data/emissions/20201116162934.032056.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201116213013.033741.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201227160831.998469.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201117073004.695373.tar.gz',
    '/home/psantos/ILU/ILU-RL/data/emissions/20201128210244.601379.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201117022052.704456.tar.gz', 
    '/home/psantos/ILU/ILU-RL/data/emissions/20201128120514.594441.tar.gz', 
]
LABEL_3 = 'DDPG'

if __name__ == '__main__':

    print('\nRUNNING analysis/stat_test.py\n')

    # Open dataframes.
    dfs_1 = []
    for exp_path in EXPS_1:

        if Path(exp_path).suffix == '.gz':
            # Compressed file (.tar.gz).

            tar = tarfile.open(exp_path)
            all_names = tar.getnames()

            # Get one of the config files.
            config_files = [x for x in all_names if Path(x).name == 'train.config']
            config_p = config_files[0]

            # Create temporary directory.
            dirpath = tempfile.mkdtemp()

            # Extract config file to temporary directory.
            tar.extract(config_p, dirpath)

            train_config = configparser.ConfigParser()
            train_config.read(dirpath + '/' + config_p)

            # Print config file.
            tls_type = train_config['train_args']['tls_type']

            # Clean temporary directory.
            shutil.rmtree(dirpath)

            exp_name = os.path.basename(exp_path)
            exp_name = exp_name.split('.')[0] + '.' + exp_name.split('.')[1]
            df = pd.read_csv(tarfile.open(exp_path).extractfile(
                                    f'{exp_name}/plots/test/{exp_name}_metrics.csv'))

            if tls_type == 'rl':
                df = df.groupby(['train_run']).mean()

            dfs_1.append(df)

        else:
            # Uncompressed file (experiment_folder).
            raise ValueError('Not implemented for uncompressed folders. Please point to a file with .tar.gz extension.')

    dfs_2 = []
    for exp_path in EXPS_2:

        if Path(exp_path).suffix == '.gz':
            # Compressed file (.tar.gz).

            tar = tarfile.open(exp_path)
            all_names = tar.getnames()

            # Get one of the config files.
            config_files = [x for x in all_names if Path(x).name == 'train.config']
            config_p = config_files[0]

            # Create temporary directory.
            dirpath = tempfile.mkdtemp()

            # Extract config file to temporary directory.
            tar.extract(config_p, dirpath)

            train_config = configparser.ConfigParser()
            train_config.read(dirpath + '/' + config_p)

            # Print config file.
            tls_type = train_config['train_args']['tls_type']

            # Clean temporary directory.
            shutil.rmtree(dirpath)

            exp_name = os.path.basename(exp_path)
            exp_name = exp_name.split('.')[0] + '.' + exp_name.split('.')[1]
            df = pd.read_csv(tarfile.open(exp_path).extractfile(
                                    f'{exp_name}/plots/test/{exp_name}_metrics.csv'))

            if tls_type == 'rl':
                df = df.groupby(['train_run']).mean()

            dfs_2.append(df)

        else:
            # Uncompressed file (experiment_folder).
            raise ValueError('Not implemented for uncompressed folders. Please point to a file with .tar.gz extension.')

    dfs_3 = []
    for exp_path in EXPS_3:

        if Path(exp_path).suffix == '.gz':
            # Compressed file (.tar.gz).

            tar = tarfile.open(exp_path)
            all_names = tar.getnames()

            # Get one of the config files.
            config_files = [x for x in all_names if Path(x).name == 'train.config']
            config_p = config_files[0]

            # Create temporary directory.
            dirpath = tempfile.mkdtemp()

            # Extract config file to temporary directory.
            tar.extract(config_p, dirpath)

            train_config = configparser.ConfigParser()
            train_config.read(dirpath + '/' + config_p)

            # Print config file.
            tls_type = train_config['train_args']['tls_type']

            # Clean temporary directory.
            shutil.rmtree(dirpath)

            exp_name = os.path.basename(exp_path)
            exp_name = exp_name.split('.')[0] + '.' + exp_name.split('.')[1]
            df = pd.read_csv(tarfile.open(exp_path).extractfile(
                                    f'{exp_name}/plots/test/{exp_name}_metrics.csv'))

            if tls_type == 'rl':
                df = df.groupby(['train_run']).mean()

            dfs_3.append(df)

        else:
            # Uncompressed file (experiment_folder).
            raise ValueError('Not implemented for uncompressed folders. Please point to a file with .tar.gz extension.')


    data_1 = []
    for df in dfs_1:
        data_1.extend(df['travel_time'].tolist())
    print(data_1)
    print(len(data_1))

    data_2 = []
    for df in dfs_2:
        data_2.extend(df['travel_time'].tolist())
    print(data_2)
    print(len(data_2))

    data_3 = []
    for df in dfs_3:
        data_3.extend(df['travel_time'].tolist())
    print(data_3)
    print(len(data_3))

    print(f'\nANOVA test: {stats.f_oneway(data_1, data_2, data_3)}')

    data = data_1 + data_2 + data_3
    groups = [LABEL_1 for _ in range(len(data_1))] + [LABEL_2 for _ in range(len(data_2))] + \
         [LABEL_3 for _ in range(len(data_3))]
    print('\nTukeyHSD:', pairwise_tukeyhsd(data, groups))

    # Non-parametric test.
    print('\nKruskal (non-parametric) test:', stats.kruskal(data_1, data_2, data_3))

    # Post-hoc non-parametric comparisons.
    data = [data_1, data_2, data_3]
    print(sp.posthoc_conover(data, p_adjust = 'holm'))