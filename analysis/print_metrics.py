import os
import re
import json
import tarfile
import pandas as pd
import argparse
from pathlib import Path
import configparser
import tempfile
import shutil

CSVS_TO_PRINT = ['cumulative_reward.csv',
                 'speed_congested_stats.csv',
                 'speed_free_flow_stats.csv',
                 'speed_stats.csv',
                 'velocity_congested_stats.csv',
                 'velocity_free_flow_stats.csv',
                 'velocity_stats.csv',
                 'stops_congested_stats.csv',
                 'stops_free_flow_stats.csv',
                 'stops_stats.csv',
                 'throughput_stats.csv',
                 'travel_time_congested_stats.csv',
                 'travel_time_free_flow_stats.csv',
                 'travel_time_stats.csv',
                 'waiting_time_congested_stats.csv',
                 'waiting_time_free_flow_stats.csv',
                 'waiting_time_stats.csv']

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            Utility script to print the metrics of a given experiment (from compressed folder).
        """
    )
    parser.add_argument('--experiment_path', help='Experiment path (compressed folder).', required=True)

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/print_metrics.py):')
    print('\tExperiment: {0}\n'.format(args.experiment_path))

def main():

    print('\nRUNNING analysis/print_metrics.py\n')

    args = get_arguments()
    print_arguments(args)
    print('\n')
    
    exp_path = args.experiment_path

    if Path(exp_path).suffix != '.gz':
        raise ValueError('Expected compressed (.tar.gz) input file.')

    tar = tarfile.open(exp_path)
    all_names = tar.getnames()

    # Get one of the demands.json files.
    demands_files = [x for x in all_names if Path(x).name == 'demands.json']

    print('demands.json:')
    json_file = json.load(tar.extractfile(demands_files[0]))
    print(json_file)

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
    for section in train_config.sections():
        print('\n')
        print(section + ':')
        print(dict(train_config[section]))

    # Clean temporary directory.
    shutil.rmtree(dirpath)

    # Print csv files.
    filtered_csvs = [x for x in all_names if Path(x).name in CSVS_TO_PRINT]

    for csv_p in filtered_csvs:
        df = pd.read_csv(tar.extractfile(csv_p))
        print('\n')
        print(Path(csv_p).name)
        print(df)

    tar.close()

if __name__ == "__main__":
    main()