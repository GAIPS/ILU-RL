#!/bin/bash
# python jobs/train.py | xargs python jobs/rollouts.py | xargs python jobs/test.py | xargs python analysis/rollouts.py
git checkout feature/pressure
python jobs/run.py

git checkout feature/speed-score
python jobs/run.py

git checkout feature/develop
python jobs/run.py

git checkout feature/pressure
python jobs/run.py

git checkout feature/speed-score
python jobs/run.py
