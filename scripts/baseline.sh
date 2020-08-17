#!/bin/bash
sed -i 's/network = grid/network \= grid_6/'  config/train.config
python jobs/baseline.py max_pressure

sed -i 's/network = grid_6/network \= grid_12/'  config/train.config
python jobs/baseline.py max_pressure

sed -i 's/network = grid_12/network \= grid_21/'  config/train.config
python jobs/baseline.py max_pressure
