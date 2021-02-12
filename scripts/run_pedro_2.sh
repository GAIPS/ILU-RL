#!/bin/bash
#
# This scripts switches features and state spaces
#

# A) EXPERIMENT: PRESSURE
python jobs/run.py

# B) EXPERIMENT: MIN QUEUE
sed -i "s/features = ('average_pressure',)/# features = ('average_pressure',)/"  config/train.config
sed -i "s/reward = 'reward_min_average_pressure'/# reward = 'reward_min_average_pressure'/"  config/train.config
sed -i "s/# features = ('queue',)/features = ('queue',)/"  config/train.config
sed -i "s/# reward = 'reward_min_queue'/reward = 'reward_min_queue'/"  config/train.config
python jobs/run.py
