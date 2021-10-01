import os
import json
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
import configparser

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

from analysis.utils import str2bool, get_emissions, get_vehicles, get_throughput
from ilurl.networks.cityflow import CityflowNetwork as Network

plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0

CONGESTED_INTERVAL = [28800.0, 32400.0] # 08h00 - 09h00
FREE_FLOW_INTERVAL = [79200.0, 82800.0] # 22h00 - 23h00


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script creates evaluation plots, given an experiment folder path.
            (To be used with RL-algorithms)
        """
    )
    parser.add_argument('experiment_root_folder', type=str, nargs='?',
                        help='Experiment root folder.')

    return parser.parse_args()

def print_arguments(args):

    print('Arguments (analysis/test_plots.py):')
    print('\tExperiment root folder: {0}\n'.format(args.experiment_root_folder))

def get_lanes_lengths(train_args):

    network_args = {
        'network_id': train_args['network'],
        'horizon': int(train_args['experiment_time']),
        'demand_type': train_args['demand_type'],
        'demand_mode': train_args['demand_mode'],
        'tls_type': train_args['tls_type']
    }
    network = Network(**network_args)

    lanes_lengths = {
        (edge['id'], int(lane['index'])): float(lane['length'])
        for edge in network.edges
        for lane in sorted(edge['lanes'], key=lambda x: int(x['index']))
    }
    return lanes_lengths

def get_length(row, lanes_lengths):
    *edge, lid =  row['lane'].split('_')
    eid = '_'.join(edge)
    lid = int(lid)
    return lanes_lengths.get((eid, lid), 0)

def main(experiment_root_folder=None):

    print('\nRUNNING analysis/test_plots.py\n')

    if not experiment_root_folder:
        args = get_arguments()
        print_arguments(args)
        experiment_root_folder = args.experiment_root_folder

    # Prepare output folder.
    output_folder_path = os.path.join(experiment_root_folder, 'plots/test')
    print('Output folder: {0}\n'.format(output_folder_path))
    os.makedirs(output_folder_path, exist_ok=True)

    # Get cycle length from tls_config.json file.
    config_files = list(Path(experiment_root_folder).rglob('tls_config.json'))
    with config_files[0].open('r') as f:
        json_file = json.load(f)
    cycle_time = json_file['rl']['cycle_time']

    # Get all *.csv files from experiment root folder.
    csv_files = [str(p) for p in list(Path(experiment_root_folder).rglob('*-emission.csv'))]
    print('Number of csv files found: {0}'.format(len(csv_files)))

    # Get agent_type and demand_type.
    train_config_path = list(Path(experiment_root_folder).rglob('train.config'))[0]
    train_config = configparser.ConfigParser()
    train_config.read(train_config_path)

    agent_type = train_config['agent_type']['agent_type']
    demand_type = train_config['train_args']['demand_type']

    vehicles_appended = []
    throughputs = []
    global_throughputs = []

    mean_values_per_eval = []

    lanes_lengths = get_lanes_lengths(train_config['train_args'])
    def fn(x):
        return get_length(x, lanes_lengths)

    for idx, csv_file in enumerate(csv_files):

        print('Processing CSV file: {0}'.format(csv_file))

        # Load CSV data.
        df_csv = get_emissions(csv_file)
        df_csv['length'] = df_csv.apply(fn, axis=1)

        df_per_vehicle = get_vehicles(df_csv, str(idx))

        df_per_vehicle_mean = df_per_vehicle.mean()

        if demand_type not in ('constant',):

            # Congested regime.
            df_congested_period = df_per_vehicle[(df_per_vehicle['finish'] > CONGESTED_INTERVAL[0]) \
                                                    & (df_per_vehicle['finish'] < CONGESTED_INTERVAL[1])]
            df_congested_period_mean = df_congested_period.mean()

            # Free-flow.
            df_free_flow_period = df_per_vehicle[(df_per_vehicle['finish'] > FREE_FLOW_INTERVAL[0]) \
                                                    & (df_per_vehicle['finish'] < FREE_FLOW_INTERVAL[1])]
            df_free_flow_period_mean = df_free_flow_period.mean()

            mean_values_per_eval.append({'train_run': Path(csv_file).parts[-4],
                                        'speed': df_per_vehicle_mean['speed'],
                                        'velocity': df_per_vehicle_mean['velocity'],
                                        'stops': df_per_vehicle_mean['stops'],
                                        'waiting_time': df_per_vehicle_mean['waiting'],
                                        'travel_time': df_per_vehicle_mean['total'],
                                        'speed_congested': df_congested_period_mean['speed'],
                                        'velocity_congested': df_congested_period_mean['velocity'],
                                        'stops_congested': df_congested_period_mean['stops'],
                                        'waiting_time_congested': df_congested_period_mean['waiting'],
                                        'travel_time_congested': df_congested_period_mean['total'],
                                        'speed_free_flow': df_free_flow_period_mean['speed'],
                                        'velocity_free_flow': df_free_flow_period_mean['velocity'],
                                        'stops_free_flow': df_free_flow_period_mean['stops'],
                                        'waiting_time_free_flow': df_free_flow_period_mean['waiting'],
                                        'travel_time_free_flow': df_free_flow_period_mean['total'],
                                        'throughput': len(df_per_vehicle)})
        else:
            mean_values_per_eval.append({'train_run': Path(csv_file).parts[-4],
                                        'speed': df_per_vehicle_mean['speed'],
                                        'velocity': df_per_vehicle_mean['velocity'],
                                        'stops': df_per_vehicle_mean['stops'],
                                        'waiting_time': df_per_vehicle_mean['waiting'],
                                        'travel_time': df_per_vehicle_mean['total'],
                                        'throughput': len(df_per_vehicle)})
                


        vehicles_appended.append(df_per_vehicle)

        df_throughput = get_throughput(df_csv)
        throughputs.append(df_throughput)
        global_throughputs.append(len(df_per_vehicle))

    df_vehicles_appended = pd.concat(vehicles_appended)
    df_throughputs_appended = pd.concat(throughputs)

    print(df_vehicles_appended.shape)
    print(df_throughputs_appended.shape)

    # Write mean values per eval into a csv file.
    df_mean_metrics_per_eval = pd.DataFrame(mean_values_per_eval)
    if demand_type not in ('constant',):
        cols = ["train_run", "speed", "velocity", "stops", "waiting_time", "travel_time", "throughput",
                "speed_congested", "velocity_congested", "stops_congested", "waiting_time_congested", "travel_time_congested",
                "speed_free_flow", "velocity_free_flow", "stops_free_flow", "waiting_time_free_flow", "travel_time_free_flow"]
    else:
        cols = ["train_run", "speed", "velocity", "stops", "waiting_time",
                "travel_time", "throughput"]

    df_mean_metrics_per_eval.to_csv('{0}/{1}_metrics.csv'.format(
                                            output_folder_path,
                                            Path(experiment_root_folder).parts[-1]
                                    ),
                                    float_format='%.3f',
                                    columns=cols)

    """
        Waiting time stats.
    """
    # Describe waiting time.
    print('Waiting time:')
    df_stats = df_vehicles_appended['waiting'].describe()
    df_stats.to_csv('{0}/waiting_time_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    # Histogram and KDE.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # plt.hist(df_vehicles_appended['waiting'], density=True)

    kde = stats.gaussian_kde(df_vehicles_appended['waiting'])
    kde_x = np.linspace(df_vehicles_appended['waiting'].min(), df_vehicles_appended['waiting'].max(), 1000)
    kde_y = kde(kde_x)
    plt.plot(kde_x, kde_y, linewidth=3)

    # Store data in dataframe for further materialization.
    waiting_time_hist_kde = pd.DataFrame()
    waiting_time_hist_kde['x'] = kde_x
    waiting_time_hist_kde['y'] = kde_y

    plt.xlabel('Waiting time (s)')
    plt.ylabel('Density')
    # plt.title('Waiting time')
    plt.savefig('{0}/waiting_time_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/waiting_time_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Travel time stats.
    """
    # Describe travel time.
    print('Travel time:')
    df_stats = df_vehicles_appended['total'].describe()
    df_stats.to_csv('{0}/travel_time_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    # Histogram and KDE.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # plt.hist(df_vehicles_appended['total'], density=True)

    kde = stats.gaussian_kde(df_vehicles_appended['total'])
    kde_x = np.linspace(df_vehicles_appended['total'].min(), df_vehicles_appended['total'].max(), 1000)
    kde_y = kde(kde_x)
    plt.plot(kde_x, kde_y, linewidth=3)

    # Store data in dataframe for further materialization.
    travel_time_hist_kde = pd.DataFrame()
    travel_time_hist_kde['x'] = kde_x
    travel_time_hist_kde['y'] = kde_y

    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    # plt.title('Travel time')
    plt.savefig('{0}/travel_time_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/travel_time_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Speed stats.
    """
    # Describe vehicles' speed.
    print('Speed:')
    df_stats = df_vehicles_appended['speed'].describe()
    df_stats.to_csv('{0}/speed_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    # Histogram and KDE.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # plt.hist(df_vehicles_appended['speed'], density=True)

    kde = stats.gaussian_kde(df_vehicles_appended['speed'])
    kde_x = np.linspace(df_vehicles_appended['speed'].min(), df_vehicles_appended['speed'].max(), 1000)
    kde_y = kde(kde_x)
    plt.plot(kde_x, kde_y, linewidth=3)

    # Store data in dataframe for further materialization.
    speed_hist_kde = pd.DataFrame()
    speed_hist_kde['x'] = kde_x
    speed_hist_kde['y'] = kde_y

    plt.xlabel('Speed (m/s)')
    plt.ylabel('Density')
    # plt.title('Vehicles\' speed')
    plt.savefig('{0}/speeds_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/speeds_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Velocity stats.
    """
    # Describe vehicles' velocity.
    print('Velocity:')
    df_stats = df_vehicles_appended['velocity'].describe()
    df_stats.to_csv('{0}/velocity_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    # Histogram and KDE.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    kde = stats.gaussian_kde(df_vehicles_appended['velocity'])
    kde_x = np.linspace(df_vehicles_appended['velocity'].min(), df_vehicles_appended['velocity'].max(), 1000)
    kde_y = kde(kde_x)
    plt.plot(kde_x, kde_y, linewidth=3)

    # Store data in dataframe for further materialization.
    velocity_hist_kde = pd.DataFrame()
    velocity_hist_kde['x'] = kde_x
    velocity_hist_kde['y'] = kde_y

    plt.xlabel('Speed (m/s)')
    plt.ylabel('Density')

    plt.savefig('{0}/velocity_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/velocity_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Stops stats.
    """
    # Describe the number of stops.
    print('Stops:')
    df_stats = df_vehicles_appended['stops'].describe()
    df_stats.to_csv('{0}/stops_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    # Histogram and KDE.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    counts = df_vehicles_appended['stops'].value_counts(normalize=True)

    plt.bar(list(counts.index), counts.values)

    # Store data in dataframe for further materialization.
    stops_hist_kde = pd.DataFrame()
    stops_hist_kde['x'] = list(counts.index)
    stops_hist_kde['y'] = counts.values

    plt.xlabel('Number of stops')
    plt.ylabel('Density')

    plt.savefig('{0}/stops_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/stops_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Throughput stats.
        (For the entire rollout)
    """
    print('Throughput:')
    df_stats = pd.DataFrame(global_throughputs).describe()
    df_stats.to_csv('{0}/throughput_stats.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)
    print(df_stats)
    print('\n')

    if demand_type not in ('constant',):

        # Filter data by congested hour interval.
        df_vehicles_appended_congested = df_vehicles_appended[(df_vehicles_appended['finish'] > CONGESTED_INTERVAL[0]) \
                                                            & (df_vehicles_appended['finish'] < CONGESTED_INTERVAL[1])]

        """
            Waiting time stats (congested).
        """
        # Describe waiting time.
        print('-'*25)
        print('Waiting time (congested):')
        df_stats = df_vehicles_appended_congested['waiting'].describe()
        df_stats.to_csv('{0}/waiting_time_congested_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        kde = stats.gaussian_kde(df_vehicles_appended_congested['waiting'])
        kde_x = np.linspace(df_vehicles_appended_congested['waiting'].min(),
                        df_vehicles_appended_congested['waiting'].max(), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, linewidth=3)

        # Store data in dataframe for further materialization.
        waiting_time_congested_hist_kde = pd.DataFrame()
        waiting_time_congested_hist_kde['x'] = kde_x
        waiting_time_congested_hist_kde['y'] = kde_y

        plt.xlabel('Waiting time (s)')
        plt.ylabel('Density')
        # plt.title('Waiting time (congested)')
        plt.savefig('{0}/waiting_time_congested_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/waiting_time_congested_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            Travel time stats (congested).
        """
        # Describe travel time.
        print('Travel time (congested):')
        df_stats = df_vehicles_appended_congested['total'].describe()
        df_stats.to_csv('{0}/travel_time_congested_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        kde = stats.gaussian_kde(df_vehicles_appended_congested['total'])
        kde_x = np.linspace(df_vehicles_appended_congested['total'].min(),
                        df_vehicles_appended_congested['total'].max(), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, linewidth=3)

        # Store data in dataframe for further materialization.
        travel_time_congested_hist_kde = pd.DataFrame()
        travel_time_congested_hist_kde['x'] = kde_x
        travel_time_congested_hist_kde['y'] = kde_y

        plt.xlabel('Travel time (s)')
        plt.ylabel('Density')
        # plt.title('Travel time (congested)')
        plt.savefig('{0}/travel_time_congested_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/travel_time_congested_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            Speed stats (congested).
        """
        # Describe vehicles' speed.
        print('Speed (congested):')
        df_stats = df_vehicles_appended_congested['speed'].describe()
        df_stats.to_csv('{0}/speed_congested_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        kde = stats.gaussian_kde(df_vehicles_appended_congested['speed'])
        kde_x = np.linspace(df_vehicles_appended_congested['speed'].min(),
                        df_vehicles_appended_congested['speed'].max(), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, linewidth=3)

        # Store data in dataframe for further materialization.
        speed_congested_hist_kde = pd.DataFrame()
        speed_congested_hist_kde['x'] = kde_x
        speed_congested_hist_kde['y'] = kde_y

        plt.xlabel('Speed (m/s)')
        plt.ylabel('Density')
        # plt.title('Vehicles\' speed (congested)')
        plt.savefig('{0}/speeds_congested_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/speeds_congested_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            Velocity stats (congested).
        """
        # Describe vehicles' velocity.
        print('Velocity (congested):')
        df_stats = df_vehicles_appended_congested['velocity'].describe()
        df_stats.to_csv('{0}/velocity_congested_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        kde = stats.gaussian_kde(df_vehicles_appended_congested['velocity'])
        kde_x = np.linspace(df_vehicles_appended_congested['velocity'].min(),
                        df_vehicles_appended_congested['velocity'].max(), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, linewidth=3)

        # Store data in dataframe for further materialization.
        velocity_congested_hist_kde = pd.DataFrame()
        velocity_congested_hist_kde['x'] = kde_x
        velocity_congested_hist_kde['y'] = kde_y

        plt.xlabel('Speed (m/s)')
        plt.ylabel('Density')

        plt.savefig('{0}/velocity_congested_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/velocity_congested_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            Stops stats (congested).
        """
        # Describe the number of stops.
        print('Stops (congested):')
        df_stats = df_vehicles_appended_congested['stops'].describe()
        df_stats.to_csv('{0}/stops_congested_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        counts = df_vehicles_appended_congested['stops'].value_counts(normalize=True)

        plt.bar(list(counts.index), counts.values)

        # Store data in dataframe for further materialization.
        stops_congested_hist_kde = pd.DataFrame()
        stops_congested_hist_kde['x'] = list(counts.index)
        stops_congested_hist_kde['y'] = counts.values

        plt.xlabel('Number of stops')
        plt.ylabel('Density')

        plt.savefig('{0}/stops_congested_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/stops_congested_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        print('-'*25)
        # Filter data by free-flow hour interval.
        df_vehicles_appended_free_flow = df_vehicles_appended[(df_vehicles_appended['finish'] > FREE_FLOW_INTERVAL[0]) \
                                                            & (df_vehicles_appended['finish'] < FREE_FLOW_INTERVAL[1])]

        """
            Waiting time stats (free-flow).
        """
        # Describe waiting time.
        print('Waiting time (free-flow):')
        df_stats = df_vehicles_appended_free_flow['waiting'].describe()
        df_stats.to_csv('{0}/waiting_time_free_flow_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        kde = stats.gaussian_kde(df_vehicles_appended_free_flow['waiting'])
        kde_x = np.linspace(df_vehicles_appended_free_flow['waiting'].min(),
                        df_vehicles_appended_free_flow['waiting'].max(), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, linewidth=3)

        # Store data in dataframe for further materialization.
        waiting_time_free_flow_hist_kde = pd.DataFrame()
        waiting_time_free_flow_hist_kde['x'] = kde_x
        waiting_time_free_flow_hist_kde['y'] = kde_y

        plt.xlabel('Waiting time (s)')
        plt.ylabel('Density')
        # plt.title('Waiting time (Free-flow)')
        plt.savefig('{0}/waiting_time_free_flow_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/waiting_time_free_flow_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            Travel time stats (free-flow).
        """
        # Describe travel time.
        print('Travel time (free-flow):')
        df_stats = df_vehicles_appended_free_flow['total'].describe()
        df_stats.to_csv('{0}/travel_time_free_flow_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        kde = stats.gaussian_kde(df_vehicles_appended_free_flow['total'])
        kde_x = np.linspace(df_vehicles_appended_free_flow['total'].min(),
                        df_vehicles_appended_free_flow['total'].max(), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, linewidth=3)

        # Store data in dataframe for further materialization.
        travel_time_free_flow_hist_kde = pd.DataFrame()
        travel_time_free_flow_hist_kde['x'] = kde_x
        travel_time_free_flow_hist_kde['y'] = kde_y

        plt.xlabel('Travel time (s)')
        plt.ylabel('Density')
        # plt.title('Travel time (Free-flow)')
        plt.savefig('{0}/travel_time_free_flow_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/travel_time_free_flow_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            Speed stats (free-flow).
        """
        # Describe vehicles' speed.
        print('Speed (free-flow):')
        df_stats = df_vehicles_appended_free_flow['speed'].describe()
        df_stats.to_csv('{0}/speed_free_flow_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        kde = stats.gaussian_kde(df_vehicles_appended_free_flow['speed'])
        kde_x = np.linspace(df_vehicles_appended_free_flow['speed'].min(),
                        df_vehicles_appended_free_flow['speed'].max(), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, linewidth=3)

        # Store data in dataframe for further materialization.
        speed_free_flow_hist_kde = pd.DataFrame()
        speed_free_flow_hist_kde['x'] = kde_x
        speed_free_flow_hist_kde['y'] = kde_y

        plt.xlabel('Speed (m/s)')
        plt.ylabel('Density')
        # plt.title('Vehicles\' speed (Free-flow)')
        plt.savefig('{0}/speeds_free_flow_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/speeds_free_flow_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            Velocity stats (free-flow).
        """
        # Describe vehicles' velocity.
        print('Velocity (free-flow):')
        df_stats = df_vehicles_appended_free_flow['velocity'].describe()
        df_stats.to_csv('{0}/velocity_free_flow_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        kde = stats.gaussian_kde(df_vehicles_appended_free_flow['velocity'])
        kde_x = np.linspace(df_vehicles_appended_free_flow['velocity'].min(),
                        df_vehicles_appended_free_flow['velocity'].max(), 1000)
        kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, linewidth=3)

        # Store data in dataframe for further materialization.
        velocity_free_flow_hist_kde = pd.DataFrame()
        velocity_free_flow_hist_kde['x'] = kde_x
        velocity_free_flow_hist_kde['y'] = kde_y

        plt.xlabel('Speed (m/s)')
        plt.ylabel('Density')

        plt.savefig('{0}/velocity_free_flow_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/velocity_free_flow_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        """
            Stops stats (free-flow).
        """
        # Describe the number of stops.
        print('Stops (free-flow):')
        df_stats = df_vehicles_appended_free_flow['stops'].describe()
        df_stats.to_csv('{0}/stops_free_flow_stats.csv'.format(output_folder_path),
                        float_format='%.3f', header=False)
        print(df_stats)
        print('\n')

        # Histogram and KDE.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        counts = df_vehicles_appended_free_flow['stops'].value_counts(normalize=True)

        plt.bar(list(counts.index), counts.values)

        # Store data in dataframe for further materialization.
        stops_free_flow_hist_kde = pd.DataFrame()
        stops_free_flow_hist_kde['x'] = list(counts.index)
        stops_free_flow_hist_kde['y'] = counts.values

        plt.xlabel('Number of stops')
        plt.ylabel('Density')

        plt.savefig('{0}/stops_free_flow_hist.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/stops_free_flow_hist.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Aggregate results per cycle.
    intervals = np.arange(0, df_vehicles_appended['finish'].max(), cycle_time)
    df_per_cycle = df_vehicles_appended.groupby(pd.cut(df_vehicles_appended["finish"], intervals)).mean()

    """
        Waiting time per cycle.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = df_per_cycle['waiting'].values
    X = np.linspace(1, len(Y), len(Y))

    # Store data in dataframe for further materialization.
    waiting_time_per_cycle = pd.DataFrame()
    waiting_time_per_cycle['x'] = X
    waiting_time_per_cycle['y'] = Y

    plt.plot(X,Y)

    plt.xlabel('Cycle')
    plt.ylabel('Average waiting time (s)')
    # plt.title('Waiting time')
    plt.savefig('{0}/waiting_time.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/waiting_time.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Travel time per cycle.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = df_per_cycle['total'].values
    X = np.linspace(1, len(Y), len(Y))

    # Store data in dataframe for further materialization.
    travel_time_per_cycle = pd.DataFrame()
    travel_time_per_cycle['x'] = X
    travel_time_per_cycle['y'] = Y

    plt.plot(X,Y)

    plt.xlabel('Cycle')
    plt.ylabel('Average travel time (s)')
    # plt.title('Travel time')
    plt.savefig('{0}/travel_time.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/travel_time.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Throughput per cycle.
        TODO: Enable throughput computation
    """
    # # Throughput per cycle.
    # fig = plt.figure()
    # fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # import ipdb; ipdb.set_trace()
    # intervals = np.arange(0, df_throughputs_appended['time'].max(), cycle_time)
    # df = df_throughputs_appended.groupby(pd.cut(df_throughputs_appended["time"], intervals)).count()

    # Y = df['time'].values
    # X = np.linspace(1, len(Y), len(Y))

    # # Store data in dataframe for further materialization.
    # throughput_per_cycle = pd.DataFrame()
    # throughput_per_cycle['x'] = X
    # throughput_per_cycle['y'] = Y

    # plt.plot(X,Y)

    # plt.xlabel('Cycle')
    # plt.ylabel('Number of vehicles')
    # # plt.title('Throughput')

    # plt.savefig('{0}/throughput.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    # plt.savefig('{0}/throughput.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    # plt.close()

    # Get test eval json file from experiment root folder.
    json_file = Path(experiment_root_folder) / 'rollouts_test.json'
    print('JSON file path: {0}\n'.format(json_file))

    # Load JSON data.
    with open(json_file) as f:
        json_data = json.load(f)

    id = str(json_data['id'][0])

    """
        Rewards per intersection (per cycle).
    """
    dfs_r = [pd.DataFrame(r) for r in json_data['rewards'][id]]

    df_concat = pd.concat(dfs_r)

    by_row_index = df_concat.groupby(df_concat.index)
    df_rewards = by_row_index.mean()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for col in df_rewards.columns:
        plt.plot(df_rewards[col].rolling(window=40).mean(), label=col)

    plt.xlabel('Cycle')
    plt.ylabel('Reward')
    # plt.title('Rewards per intersection')
    plt.legend()

    plt.savefig('{0}/rewards_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/rewards_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    plt.close()

    """
        Total rewards (per cycle).
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y) 

    plt.plot(df_rewards.sum(axis=1))

    plt.xlabel('Cycle')
    plt.ylabel('Reward')
    # plt.title('Cumulative reward')

    plt.savefig('{0}/total_reward.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/total_reward.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    
    plt.close()

    total_reward = df_rewards.to_numpy().sum()

    # Describe total system cumulative reward.
    pd.DataFrame([total_reward]).to_csv('{0}/cumulative_reward.csv'.format(output_folder_path),
                    float_format='%.3f', header=False)

    """
        Actions per intersection (per cycle).

        WARNING: This might require different processing here. As an example,
            the actions taken by the DQN actions (discrete action agent)
            differ from the ones taken by the DDPG agent (continuous action
            agent).
    """
    if agent_type in ('DDPG', 'MPO'):
        # Continuous action-schema.
        # TODO: This only works for two-phased intersections.
        dfs_a = [pd.DataFrame([{i: round(a[0], 4) for (i, a) in t.items()}
                                for t in run])
                                    for run in json_data['actions'][id]]
        df_concat = pd.concat(dfs_a)

        by_row_index = df_concat.groupby(df_concat.index)
        df_actions = by_row_index.mean()

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        # window_size = min(len(df_actions)-1, 40)

        for col in df_actions.columns:
            plt.plot(df_actions[col], label=col) # .rolling(window=window_size).mean()

        plt.xlabel('Cycle')
        plt.ylabel('Action (phase-1 allocation)')
        # plt.title('Actions per intersection')
        plt.legend()

        plt.ylim(0.0,1.0)

        plt.savefig('{0}/actions_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/actions_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

        plt.close()

    else:
        # Discrete action-schema.
        dfs_a = [pd.DataFrame(run) for run in json_data['actions'][id]]

        df_concat = pd.concat(dfs_a)

        by_row_index = df_concat.groupby(df_concat.index)
        df_actions = by_row_index.mean()

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for col in df_actions.columns:
            plt.plot(df_actions[col].rolling(window=40).mean(), label=col)

        plt.xlabel('Cycle')
        plt.ylabel('Phase')
        # plt.title('Actions per intersection')
        plt.legend()

        plt.ylim(-0.2,1.2)
        plt.yticks(ticks=[0,1], labels=['0', '1'])

        plt.savefig('{0}/actions_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/actions_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

        plt.close()

    """
           Actions

           WARNING: This might require different processing here. As an example,
               the actions taken by the DQN actions (discrete action agent)
               differ from the ones taken by the DDPG agent (continuous action
               agent).
       """
    if agent_type in ('DDPG', 'MPO'):
        # Continuous action-schema.
        # TODO: This only works for two-phased intersections.
        dfs_a = [pd.DataFrame([{i: round(a[0], 4) for (i, a) in t.items()}
                               for t in run])
                 for run in json_data['actions'][id]]
        df_concat = pd.concat(dfs_a)

        by_row_index = df_concat.groupby(df_concat.index)
        df_actions = by_row_index.mean()

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        # window_size = min(len(df_actions)-1, 40)

        for col in df_actions.columns:
            plt.plot(df_actions[col], label=col)  # .rolling(window=window_size).mean()

        plt.xlabel('Cycle')
        plt.ylabel('Action (phase-1 allocation)')
        # plt.title('Actions per intersection')
        plt.legend()

        plt.ylim(0.0, 1.0)

        plt.savefig('{0}/actions_per_intersection.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/actions_per_intersection.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

        plt.close()

    else:
        # Discrete action-schema.
        dfs_a = [pd.DataFrame(run) for run in json_data['actions'][id]]
        df_concat = pd.concat(dfs_a)

        by_row_index = df_concat.groupby(df_concat.index)
        df_actions = by_row_index.mean()
        df_actions = df_actions.mean(axis=1)

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        plt.plot(df_actions.rolling(window=40).mean())

        plt.xlabel('Cycle')
        plt.ylabel('Phase')
        # plt.title('Actions per intersection')
        plt.legend()

        plt.ylim(-0.2, 1.2)
        plt.yticks(ticks=[0, 1], labels=['0', '1'])

        plt.savefig('{0}/actions.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/actions.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

        plt.close()

    """
        Number of vehicles per cycle.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    dfs_veh = [pd.DataFrame(r) for r in json_data['vehicles'][id]]

    df_concat = pd.concat(dfs_veh)

    by_row_index = df_concat.groupby(df_concat.index)
    df_vehicles = by_row_index.mean()
    
    X = np.arange(0, len(df_vehicles))
    Y = df_vehicles

    # Store data in dataframe for further materialization.
    vehicles_per_cycle = pd.DataFrame()
    vehicles_per_cycle['x'] = X
    vehicles_per_cycle['y'] = Y

    plt.plot(X,Y)

    plt.xlabel('Cycle')
    plt.ylabel('Number of vehicles')
    # plt.title('Number of vehicles')

    plt.savefig('{0}/vehicles.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/vehicles.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    
    plt.close()

    """
        Average vehicles' velocity per cycle.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    dfs_vels = [pd.DataFrame(r) for r in json_data['velocities'][id]]

    df_concat = pd.concat(dfs_vels)

    by_row_index = df_concat.groupby(df_concat.index)
    df_velocities = by_row_index.mean()

    X = np.arange(0, len(df_velocities))
    Y = df_velocities

    # Store data in dataframe for further materialization.
    velocities_per_cycle = pd.DataFrame()
    velocities_per_cycle['x'] = X
    velocities_per_cycle['y'] = Y

    plt.plot(X,Y)

    plt.xlabel('Cycle')
    plt.ylabel('Average velocity (m/s)')
    # plt.title('Vehicles\' velocities')

    plt.savefig('{0}/velocities.pdf'.format(output_folder_path), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/velocities.png'.format(output_folder_path), bbox_inches='tight', pad_inches=0)

    plt.close()


    # Materialize processed data.
    if demand_type not in ('constant'):
        # TODO: throughput_per_cycle, keys='throughput_per_cycle'
        processed_data = pd.concat([waiting_time_hist_kde,
                                    travel_time_hist_kde,
                                    speed_hist_kde,
                                    velocity_hist_kde,
                                    stops_hist_kde,
                                    waiting_time_congested_hist_kde,
                                    travel_time_congested_hist_kde,
                                    speed_congested_hist_kde,
                                    velocity_congested_hist_kde,
                                    stops_congested_hist_kde,
                                    waiting_time_free_flow_hist_kde,
                                    travel_time_free_flow_hist_kde,
                                    speed_free_flow_hist_kde,
                                    velocity_free_flow_hist_kde,
                                    stops_free_flow_hist_kde,
                                    waiting_time_per_cycle,
                                    travel_time_per_cycle,
                                    vehicles_per_cycle,
                                    velocities_per_cycle]
                                    , keys=['waiting_time_hist_kde',
                                    'travel_time_hist_kde',
                                    'speed_hist_kde',
                                    'velocity_hist_kde',
                                    'stops_hist_kde',
                                    'waiting_time_congested_hist_kde',
                                    'travel_time_congested_hist_kde',
                                    'speed_congested_hist_kde',
                                    'velocity_congested_hist_kde',
                                    'stops_congested_hist_kde',
                                    'waiting_time_free_flow_hist_kde',
                                    'travel_time_free_flow_hist_kde',
                                    'speed_free_flow_hist_kde',
                                    'velocity_free_flow_hist_kde',
                                    'stops_free_flow_hist_kde',
                                    'waiting_time_per_cycle',
                                    'travel_time_per_cycle',
                                    'vehicles_per_cycle',
                                    'velocities_per_cycle']
                                    , axis=1)
    else:  
        processed_data = pd.concat([waiting_time_hist_kde,
                                    travel_time_hist_kde,
                                    speed_hist_kde,
                                    velocity_hist_kde,
                                    stops_hist_kde,
                                    waiting_time_per_cycle,
                                    travel_time_per_cycle,
                                    vehicles_per_cycle,
                                    velocities_per_cycle]
                                    , keys=['waiting_time_hist_kde',
                                    'travel_time_hist_kde',
                                    'speed_hist_kde',
                                    'velocity_hist_kde',
                                    'stops_hist_kde',
                                    'waiting_time_per_cycle',
                                    'travel_time_per_cycle',
                                    'vehicles_per_cycle',
                                    'velocities_per_cycle']
                                    , axis=1)

    processed_data.to_csv('{0}/processed_data.csv'.format(
                                            output_folder_path),
                                            float_format='%.6f')


if __name__ == "__main__":
    main()
