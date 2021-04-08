"""

    Python script to run full pipeline:

        0) jobs/preprocess: makes discretization bin

        1) jobs/train.py: train agent(s).

        2) analysis/train_plots.py: create training plots.

        3) jobs/rollouts.py: execute and evaluate different
                             checkpoints using rollouts.

        4) analysis/rollouts.py: create rollouts plots.

        5) jobs/rollouts.py: execute python script in test
                             mode in order to assess final
                             agent's performance.

                             (Generates *.xml files).

        6) [Convert *.xml files to *.csv files].

        7) analysis/test_plots.py: Create plots with metrics
                             for the final agent.

        8) Clean up and compress the experiment folder in
            order to optimize disk space usage.

"""
import os
import shutil
from pathlib import Path

import ilurl.loaders.parser
from ilurl.loaders.parser import config_parser
from jobs.preprocess import preprocess_batch as preprocess
from jobs.train import train_batch as train
from jobs.rollouts import rollout_batch as rollouts
from jobs.convert2csv import xml2csv

from analysis.train_plots import main as train_plots
from analysis.rollouts import main as rollouts_plots
from analysis.test_plots import main as test_plots

from ilurl.utils.decorators import safe_run

# Ignore exceptions (plots).
_ERROR_MESSAGE_TRAIN = ("ERROR: Caught an exception while "
                        "executing analysis/train_plots.py script.")
_ERROR_MESSAGE_ROLLOUTS = ("ERROR: Caught an exception while "
                           "executing analysis/rollouts_plots.py script.")
_ERROR_MESSAGE_TEST = ("ERROR: Caught an exception while "
                       "executing analysis/test_plots.py script.")
train_plots = safe_run(train_plots, error_message=_ERROR_MESSAGE_TRAIN)
rollouts_plots = safe_run(rollouts_plots, error_message=_ERROR_MESSAGE_ROLLOUTS)
test_plots = safe_run(test_plots, error_message=_ERROR_MESSAGE_TEST)

if __name__ == '__main__':

    # 0) Makes discretization bins
    categories = preprocess()

    # 1) Train agent(s).
    experiment_root_path = train(categories)

    # 2) Create train plots.
    train_plots(experiment_root_path)

    # 3) Execute rollouts.
    # eval_path = rollouts(experiment_dir=experiment_root_path)

    # 4) Create rollouts plots.
    # rollouts_plots(eval_path)

    # 5) Execute rollouts with last saved checkpoints (test).
    rollouts(test=True, experiment_dir=experiment_root_path)

    # 6) Convert .xml files to .csv files.
    xml2csv(experiment_root_path=experiment_root_path)

    # 7) Create plots with metrics plots for final agent.
    test_plots(experiment_root_path)

    # 8) Clean up and compress files.
    print('\nCleaning and compressing files...\n')
    experiment_root_path = Path(experiment_root_path)
    for csv_path in experiment_root_path.rglob('*-emission.csv'):
        Path(csv_path).unlink()

    shutil.make_archive(experiment_root_path,
                        'gztar',
                        os.path.dirname(experiment_root_path),
                        experiment_root_path.name)
    shutil.rmtree(experiment_root_path)

    print('Experiment folder: {0}'.format(experiment_root_path))
