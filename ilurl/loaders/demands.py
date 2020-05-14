"""Path resolution for demand files"""
from os import environ
from pathlib import Path
import json

path = Path(f"{environ['ILURL_HOME']}/data/demands/")

def get_demand(demand_type=None):
    demand_path = path / 'demands.json'

    with demand_path.open(mode='r') as f:
        demand_data = json.load(f)

    if demand_type not in demand_data:
        raise ValueError(f'Missing demand_type {demand_type} in demands.json file.')

    return demand_data[demand_type]
