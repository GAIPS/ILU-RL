import os
import re
import tarfile
import pandas as pd
import argparse
from pathlib import Path

CSVS_TO_PRINT = ['cumulative_reward.csv',
                 'speed_congested_stats.csv',
                 'speed_free_flow_stats.csv',
                 'speed_stats.csv',
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
    
    exp_path = args.experiment_path

    if Path(exp_path).suffix != '.gz':
        raise ValueError('Expected compressed (.tar.gz) input file.')

    tar = tarfile.open(exp_path)

    all_names = tar.getnames()
    filtered_csvs = [x for x in all_names if Path(x).name in CSVS_TO_PRINT]

    for csv_p in filtered_csvs:
        tar_file = tar.extractfile(csv_p)
        df = pd.read_csv(tar_file)
        
        print('\n')
        print(Path(csv_p).name)
        print(df)

if __name__ == "__main__":
    main()