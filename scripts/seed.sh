#!/bin/bash
# python jobs/train.py | xargs python jobs/rollouts.py | xargs python jobs/test.py | xargs python analysis/rollouts.py
# add 100
sed -i 's/\[50, 60, 70, 80, 90, 100, 110, 120, 130, 140\]/\[150, 160, 170, 180, 190, 200, 210, 220, 230, 240\]/g' config/run.config
git add config/run.config scripts/seed.sh
git commit 'Set config INTER., QL, 50k, 60s, 10*3, SPEED&COUNT (2)'
python jobs/run.py

