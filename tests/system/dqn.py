import os
import json
import unittest
import configparser

from models.train import main as train

ILURL_HOME = os.environ['ILURL_HOME']
CONFIG_PATH = \
    f'{ILURL_HOME}/tests/system/config'

OUTPUTS_PATH = \
    f'{ILURL_HOME}/tests/system/outputs'

SEED = 77

class TestDQN(unittest.TestCase):
    """
        System test for the DQN agent.
    """

    def test_train_dqn_speed_count(self):

        # Read train.py arguments from train.config file.
        train_config = configparser.ConfigParser()
        cfg_path = os.path.join(CONFIG_PATH, 'test_train_dqn_speed_count.config')
        train_config.read(cfg_path)

        path = train(cfg_path)

        # Load train_log.json data.
        with open(path + '/logs/train_log.json') as f:
            json_data_1 = json.load(f)

        # Load expected output.
        with open(OUTPUTS_PATH + '/test_train_dqn_speed_count.json') as f:
            json_data_2 = json.load(f)

        self.assertEqual(json_data_1, json_data_2)


    def test_train_dqn_delay(self):

        # Read train.py arguments from train.config file.
        train_config = configparser.ConfigParser()
        cfg_path = os.path.join(CONFIG_PATH, 'test_train_dqn_delay.config')
        train_config.read(cfg_path)

        path = train(cfg_path)

        # Load train_log.json data.
        with open(path + '/logs/train_log.json') as f:
            json_data_1 = json.load(f)

        # Load expected output.
        with open(OUTPUTS_PATH + '/test_train_dqn_delay.json') as f:
            json_data_2 = json.load(f)

        self.assertEqual(json_data_1, json_data_2)
