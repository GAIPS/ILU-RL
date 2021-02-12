#!/bin/bash

# Grid_6 + Constant
python jobs/baseline.py actuated
python jobs/baseline.py max_pressure
python jobs/baseline.py webster
python jobs/baseline.py random

# Grid_6 + variable
sed -i "s/demand_type = constant/demand_type = variable/"  config/train.config

python jobs/baseline.py actuated
python jobs/baseline.py max_pressure
python jobs/baseline.py webster
python jobs/baseline.py random
