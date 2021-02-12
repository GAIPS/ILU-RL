#!/bin/bash

# high + dqn + delta delay reduction.
python jobs/run.py

# low + ql + min queue
sed -i "s/# features = ('queue',)/features = ('queue',)/"  config/train.config
sed -i "s/# reward = 'reward_min_queue'/reward = 'reward_min_queue'/"  config/train.config
sed -i "s/features = ('delay', 'lag\[delay\]')/# features = ('delay', 'lag\[delay\]')/"  config/train.config
sed -i "s/reward = 'reward_max_delay_reduction'/# reward = 'reward_max_delay_reduction'/"  config/train.config
sed -i "s/discretize_state_space = False/discretize_state_space = True/"  config/train.config
sed -i "s/agent_type = DQN/agent_type = QL/"  config/train.config

sed -i 's/"1": 0.10, "2": 0.22/"1": 0.05, "2": 0.10/'  data/networks/intersection/demands.json

python jobs/run.py
