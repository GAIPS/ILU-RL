#!/bin/bash
#
# This scripts switches features and state spaces
#
# TLS_TYPE, DEMAND_TYPE,
python jobs/run.py
git commit -a -m 'CENTRALIZED, CONSTANT DEMAND, SPEEDCOUNT'

# B) EXPERIMENT: INTER, DELAY
sed -i "s/tls_type = centralized/# tls_type = centralized/"  config/train.config
sed -i "s/#tls_type = rl/tls_type = rl/"  config/train.config
python jobs/run.py
git commit -a -m 'RL, CONSTANT DEMAND, SPEEDCOUNT'
