import pandas as pd
import numpy as np

EXCLUDE_EMISSION = ['CO', 'CO2', 'HC', 'NOx', 'PMx', 'angle', 'eclass', 'electricity', 'fuel', 'noise']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_emissions(file_path, exclude_emissions=EXCLUDE_EMISSION):
    """Gets an emission file

    Parameters:
    ----------
    * file_path
    * exclude_emissions

    Return:
    ------
    * df pandas.DataFrame

    """
    df = pd.read_csv(file_path, sep=';', header=0, encoding='utf-8')

    # The token 'vehicle_' comes when using SUMOS's script
    # referece sumo/tools/xml2csv
    df.columns = [str.replace(str(name), 'vehicle_', '') for name in df.columns]
    df.columns = [str.replace(str(name), 'timestep_', '') for name in df.columns]

    df.set_index(['time'], inplace=True)

    # Drop rows where there's no vehicle
    df = df.dropna(axis=0, how='all')

    # Drop columns if needed
    if exclude_emissions is not None:
        df = df.drop(exclude_emissions, axis=1, errors='ignore')

    return df

def get_vehicles(emissions_df):
    """Returns vehicle data

    Parameters:
    ----------
    * emissions_df: pandas DataFrame
        SEE get_emission

    Usage:
    -----
    ipdb> vehs_df = get_vehicles(emissions_df)
    ipdb> vehs_df.head()
               route finish  start  wait  total
    id
    flow_00.0  route309265401#0_0   11.3    1.0   0.0   10.3
    flow_00.1  route309265401#0_0   18.4    7.1   0.0   11.3
    flow_00.2  route309265401#0_2   24.0   13.3   0.0   10.7
    flow_00.3  route309265401#0_2   29.7   19.4   0.0   10.3
    flow_00.4  route309265401#0_2   36.1   25.6   0.0   10.5
    """
    # Builds a dataframe with vehicle starts
    start_df = pd.pivot_table(
        emissions_df.reset_index(),
        index=['id', 'route'], values='time',
        aggfunc=min
    ). \
    reset_index('route'). \
    rename(columns={'time': 'start'}, inplace=False)

    # Builds a dataframe with vehicle finish
    finish_df = pd.pivot_table(
        emissions_df.reset_index(),
        index='id', values='time',
        aggfunc=max
    ).\
    rename(columns={'time': 'finish'}, inplace=False)

    # Builds a dataframe with waiting times
    wait_df = pd.pivot_table(
        emissions_df.reset_index(),
        index='id', values='waiting',
        aggfunc=max
    ).\
    rename(columns={'time': 'wait'}, inplace=False)

    speed_df = pd.pivot_table(
        emissions_df.reset_index(),
        index='id', values='speed',
        aggfunc=np.mean
    ).\
    rename(columns={'time': 'speed'}, inplace=False)

    vehs_df = start_df.join(
        finish_df, on='id', how='inner',
    ). \
    sort_values('start', inplace=False). \
    join(wait_df, on='id', how='left')

    vehs_df['total'] = vehs_df['finish'] - vehs_df['start']

    vehs_df = vehs_df.join(
        speed_df, on='id', how='inner',
    )
    return vehs_df


def get_throughput(df_emission):

    # depending on the conversion options
    # and net configurations the field
    # might change labels.
    if 'edge_id' in df_emission.columns:
        col_edge = 'edge_id'
    else:
        col_edge = 'lane'

    in_junction = df_emission[col_edge].str.startswith(':')

    df_junction = df_emission[in_junction].sort_values(by=['id', 'time'])

    df_junction = df_junction.drop_duplicates(subset='id', keep='first').reset_index()

    df_junction = df_junction[['time','id']]

    return df_junction