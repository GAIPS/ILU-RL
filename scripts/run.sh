#!/bin/bash
#
# This scripts switches features and state spaces
#
# A) EXPERIMENT: INTER, SPEED & COUNT
git commit -a -m 'Set low INTER, QL, 50k, 60s, 30, SPEED.COUNT' 
python jobs/run.py

# B) EXPERIMENT: INTER, DELAY
sed -i "s/features = ('speed', 'count')/# features = ('speed', 'count')/"  config/train.config
sed -i "s/reward = 'reward_min_speed_delta'/# reward = 'reward_min_speed_delta'/"  config/train.config

sed -i "s/# features = ('delay',)/features = ('delay',)/"  config/train.config
sed -i "s/# reward = 'reward_min_delay'/reward = 'reward_min_delay'/"  config/train.config

git commit -a -m 'Set low INTER, QL, 50k, 60s, 30, DELAY' 
python jobs/run.py


# C) EXPERIMENT: INTER, QUEUE
sed -i "s/features = ('delay',)/# features = ('delay',)/"  config/train.config
sed -i "s/reward = 'reward_min_delay'/# reward = 'reward_min_delay'/"  config/train.config

sed -i "s/# features = ('queue', 'lag\[queue\]')/features = ('queue', 'lag\[queue\]')/"  config/train.config
sed -i "s/# reward = 'reward_min_queue_squared'/reward = 'reward_min_queue_squared'/"  config/train.config
git commit -a -m 'Set low INTER, QL, 50k, 60s, 30, QUEUE' 
python jobs/run.py

# D) EXPERIMENT: INTER, WAITING TIME
sed -i "s/# features = ('waiting_time',)/features = ('waiting_time',)/"  config/train.config
sed -i "s/# reward = 'reward_min_waiting_time'/reward = 'reward_min_waiting_time'/"  config/train.config


sed -i "s/features = ('queue', 'lag\[queue\]')/# features = ('queue', 'lag\[queue\]')/"  config/train.config
sed -i "s/reward = 'reward_min_queue_squared'/# reward = 'reward_min_queue_squared'/"  config/train.config
git commit -a -m 'Set low INTER, QL, 50k, 60s, 30, WAIT' 
python jobs/run.py

# E) EXPERIMENT: INTER, SPEED SCORE
sed -i "s/# features = ('speed_score', 'count')/features = ('speed_score', 'count')/"  config/train.config
sed -i "s/# reward = 'reward_max_speed_score'/reward = 'reward_max_speed_score'/"  config/train.config

sed -i "s/features = ('waiting_time',)/# features = ('waiting_time',)/"  config/train.config
sed -i "s/reward = 'reward_min_waiting_time'/# reward = 'reward_min_waiting_time'/"  config/train.config
git commit -a -m 'Set low INTER, QL, 50k, 60s, 30, SPEED.SCORE' 
python jobs/run.py

# F) EXPERIMENT: INTER, AVERAGE PRESSURE
sed -i "s/# features = ('average_pressure',)/features = ('average_pressure',)/"  config/train.config
sed -i "s/# reward = 'reward_min_average_pressure'/reward = 'reward_min_average_pressure'/"  config/train.config


sed -i "s/features = ('speed_score', 'count')/# features = ('speed_score', 'count')/"  config/train.config
sed -i "s/reward = 'reward_max_speed_score'/# reward = 'reward_max_speed_score'/"  config/train.config
git commit -a -m 'Set low INTER, QL, 50k, 60s, 30, AVG.PRESS' 
python jobs/run.py


# G) EXPERIMENT: INTER, FLOW
sed -i "s/# features = ('flow',)/features = ('flow',)/"  config/train.config
sed -i "s/# reward = 'reward_max_flow'/reward = 'reward_max_flow'/"  config/train.config


sed -i "s/features = ('average_pressure',)/# features = ('average_pressure',)/"  config/train.config
sed -i "s/reward = 'reward_min_average_pressure'/# reward = 'reward_min_average_pressure'/"  config/train.config
git commit -a -m 'Set cyclical INTER, QL, 50k, 60s, 30, FLOW' 
python jobs/run.py

# H) EXPERIMENT: INTER, FLOW
sed -i "s/features = ('flow',)/# features = ('flow',)/"  config/train.config
sed -i "s/reward = 'reward_max_flow'/# reward = 'reward_max_flow'/"  config/train.config


sed -i "s/# features = ('delay', 'lag\[delay\]')/features = ('delay', 'lag\[delay\]')/"  config/train.config
sed -i "s/# reward = 'reward_max_delay_reduction'/reward = 'reward_max_delay_reduction'/"  config/train.config
git commit -a -m 'Set cyclical INTER, QL, 50k, 60s, 30, DDEL' 
python jobs/run.py
