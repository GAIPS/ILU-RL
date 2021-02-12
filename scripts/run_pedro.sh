#!/bin/bash
#
# This scripts switches features and state spaces
#

# A) EXPERIMENT: SPEED & COUNT
python jobs/run.py

# B) EXPERIMENT: DELAY
sed -i "s/features = ('speed', 'count')/# features = ('speed', 'count')/"  config/train.config
sed -i "s/reward = 'reward_min_speed_delta'/# reward = 'reward_min_speed_delta'/"  config/train.config
sed -i "s/# features = ('delay',)/features = ('delay',)/"  config/train.config
sed -i "s/# reward = 'reward_min_delay'/reward = 'reward_min_delay'/"  config/train.config
python jobs/run.py

# C) EXPERIMENT: QUEUE
# sed -i "s/features = ('delay',)/# features = ('delay',)/"  config/train.config
# sed -i "s/reward = 'reward_min_delay'/# reward = 'reward_min_delay'/"  config/train.config
# sed -i "s/# features = ('queue', 'lag\[queue\]')/features = ('queue', 'lag\[queue\]')/"  config/train.config
# sed -i "s/# reward = 'reward_min_queue_squared'/reward = 'reward_min_queue_squared'/"  config/train.config
# python jobs/run.py

# D) EXPERIMENT: WAITING TIME
sed -i "s/# features = ('waiting_time',)/features = ('waiting_time',)/"  config/train.config
sed -i "s/# reward = 'reward_min_waiting_time'/reward = 'reward_min_waiting_time'/"  config/train.config
sed -i "s/features = ('delay',)/# features = ('delay',)/"  config/train.config
sed -i "s/reward = 'reward_min_delay'/# reward = 'reward_min_delay'/"  config/train.config
python jobs/run.py

# E) EXPERIMENT: PRESSURE
# sed -i "s/# features = ('average_pressure',)/features = ('average_pressure',)/"  config/train.config
# sed -i "s/# reward = 'reward_min_average_pressure'/reward = 'reward_min_average_pressure'/"  config/train.config
# sed -i "s/features = ('waiting_time',)/# features = ('waiting_time',)/"  config/train.config
# sed -i "s/reward = 'reward_min_waiting_time'/# reward = 'reward_min_waiting_time'/"  config/train.config
# python jobs/run.py
