import pickle
import unittest

import numpy as np

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.params import MDPParams
from ilurl.networks.base import Network
from ilurl.utils.aux import flatten


class TestStateReward(unittest.TestCase):
    """
        Set of tests that target the implemented
        problem formulations, i.e. state and reward
        function definitions.
    """

    def setUp(self):

        network_args = {
            'network_id': 'grid',
            'horizon': 999,
            'demand_type': 'constant',
            'tls_type': 'rl'
        }
        self.network = Network(**network_args)


    def test_time_period(self):
        """
            Time period.
        """
        mdp_params = MDPParams(
                        features=('delay',),
                        reward='reward_min_delay',
                        normalize_velocities=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=3600,
                        velocity_threshold=0.1)

        self.observation_space = State(self.network, mdp_params)
        self.observation_space.reset()

        with open('tests/unit/data/grid_kernel_data.dat', "rb") as f:
            kernel_data = pickle.load(f)

        self.assertEqual(len(kernel_data), 60)

        # Fake environment interaction with state object.
        timesteps = list(range(1,60)) + [0]
        for t, data in zip(timesteps, kernel_data):
            self.observation_space.update(t, data)

        # Get state.
        state = self.observation_space.feature_map(
            categorize=mdp_params.discretize_state_space,
            flatten=True
        )

        self.assertEqual(len(state['247123161']), 3)
        self.assertEqual(len(state['247123464']), 3)
        self.assertEqual(len(state['247123468']), 3)

        # State.
        # 247123161.
        self.assertEqual(state['247123161'][0], 0)    # time variable

        # 247123464.
        self.assertEqual(state['247123464'][0], 0)    # time variable

        # 247123468.
        self.assertEqual(state['247123468'][0], 0)    # time variable

        self.observation_space.reset()

        hours = list(range(24)) + [0,1]
        for hour in hours:
            for minute in range(60):

                # Fake environment interaction with state object.
                # (60 seconds = 1 minute).
                timesteps = list(range(1,60)) + [0]
                for t, data in zip(timesteps, kernel_data):
                    self.observation_space.update(t, data)

                # Get state.
                state = self.observation_space.feature_map(
                    categorize=mdp_params.discretize_state_space,
                    flatten=True
                )

                self.assertEqual(state['247123161'][0], hour)
                self.assertEqual(state['247123464'][0], hour)
                self.assertEqual(state['247123468'][0], hour)


    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
