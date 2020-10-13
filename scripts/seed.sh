#!/bin/bash
# python jobs/train.py | xargs python jobs/rollouts.py | xargs python jobs/test.py | xargs python analysis/rollouts.py
# add 100
<<<<<<< HEAD
python jobs/run.py

sed -i 's/\[0, 10, 20, 30, 40\]/\[50, 60, 70, 80, 90\]/g' config/run.config
python jobs/run.py


sed -i 's/\[50, 60, 70, 80, 90\]/\[100, 110, 120, 130, 140\]/g' config/run.config
python jobs/run.py

sed -i 's/\[100, 110, 120, 130, 140\]/\[150, 160, 170, 180, 190\]/g' config/run.config
python jobs/run.py

sed -i 's/\[150, 160, 170, 180, 190\]/\[200, 210, 220, 230, 240\]/g' config/run.config
python jobs/run.py
=======
sed -i 's/\[50, 60, 70, 80, 90, 100, 110, 120, 130, 140\]/\[150, 160, 170, 180, 190, 200, 210, 220, 230, 240\]/g' config/run.config
git add config/run.config scripts/seed.sh
git commit 'Set config INTER., QL, 50k, 60s, 10*3, SPEED&COUNT (2)'
python jobs/run.py

>>>>>>> new-home
