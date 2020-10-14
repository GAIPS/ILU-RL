#!/bin/bash
#
# This scripts switches features and state spaces
#
# A) EXPERIMENT: INTERS., SPEED & COUNT
# git commit -a -m 'Set config INTERS., DQN, 50k, 60s, 30, SPEED.COUNT' 
# python jobs/run.py

# B) EXPERIMENT: INTERS., DELAY
# sed -i "s/features = ('speed', 'count')/# features = ('speed', 'count')/"  config/train.config
# sed -i "s/reward = 'reward_min_speed_delta'/# reward = 'reward_min_speed_delta'/"  config/train.config
# 
# sed -i "s/# features = ('delay',)/features = ('delay',)/"  config/train.config
# sed -i "s/# reward = 'reward_min_delay'/reward = 'reward_min_delay'/"  config/train.config

git commit -a -m 'Set config INTERS., DQN, 30k, 60s, 30, DELAY' 
python jobs/run.py

# D) EXPERIMENT: INTERS., WAITING TIME
sed -i "s/# features = ('waiting_time',)/features = ('waiting_time',)/"  config/train.config
sed -i "s/# reward = 'reward_min_waiting_time'/reward = 'reward_min_waiting_time'/"  config/train.config


sed -i "s/features = ('delay',)/# features = ('delay',)/"  config/train.config
sed -i "s/reward = 'reward_min_delay'/# reward = 'reward_min_delay'/"  config/train.config
git commit -a -m 'Set config INTERS., DQN, 30k, 60s, 30, WAIT' 
python jobs/run.py

