"""Path resolution for demand files"""
from os import environ
from pathlib import Path
import json

def get_demand(demand_type=None, network_id=None):

    file = Path(f"{environ['ILURL_HOME']}/data/networks/{network_id}/demands.json")

    if file.exists ():
        # Custom file.

        with file.open(mode='r') as f:
            demand_data = json.load(f)

        if demand_type not in demand_data:
            raise ValueError(f'Missing demand_type {demand_type} in demands.json file.')

        return demand_data[demand_type]

    else:
        # Default file.

        print('WARNING: Custom demands.json file missing. Loading default file from ilurl/data/demands')

        path = Path(f"{environ['ILURL_HOME']}/data/demands/")

        default_demand_path = path / 'demands.json'
        with default_demand_path.open(mode='r') as f:
            demand_data = json.load(f)

        if demand_type not in demand_data:
            raise ValueError(f'Missing demand_type {demand_type} in demands.json file.')

        return demand_data[demand_type]
