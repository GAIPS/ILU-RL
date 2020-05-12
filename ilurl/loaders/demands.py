"""Path resolution for demand files"""
from os import environ
from pathlib import Path
import json

demand_path = Path(f"{environ['ILURL_HOME']}/data/demands/")


def get_uniform(intensity=None):
    uniform_path = demand_path / 'uniform.json'

    with uniform_path.open(mode='r') as f:
        demand_data = json.load(f)

    uniform = demand_data['insertion_probabilities']
    if intensity in uniform:
        return uniform[intensity]
    return uniform



    
