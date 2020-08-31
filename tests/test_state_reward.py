import pickle
import unittest

import numpy as np

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.networks.base import Network


class TestStateReward(unittest.TestCase):

    def setUp(self):

        network_args = {
            'network_id': 'grid',
            'horizon': 999,
            'demand_type': 'constant',
            'tls_type': 'rl'
        }
        self.network = Network(**network_args)

    def test_speed_count(self):

        mdp_params = MDPParams(
                        features=('speed', 'count'),
                        reward='reward_max_speed_count',
                        normalize_state_space=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None)

        self.observation_space = State(self.network, mdp_params)
        self.observation_space.reset()
        self.reward = build_rewards(mdp_params)
    
        with open('tests/data/grid_kernel_data.dat', "rb") as f:
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

        """
        print(state)
        values_0_count = []
        values_1_count = []
        ID = '247123468'
        for t in kernel_data:
            values_0_count.extend(t[ID][0])
            values_1_count.extend(t[ID][1])
        print('count0', len(values_0_count)/60)
        print('count1', len(values_1_count)/60)

        vehs_speeds_0 = []
        vehs_speeds_1 = []
        for veh in values_0_count:
            vehs_speeds_0.append(veh.speed)
        for veh in values_1_count:
            vehs_speeds_1.append(veh.speed)

        vehs_speeds_0 = np.array(vehs_speeds_0)
        vehs_speeds_1 = np.array(vehs_speeds_1)

        print(np.sum((13.89 - vehs_speeds_0) / 13.89) / len(values_0_count))
        print(np.sum((13.89 - vehs_speeds_1) / 13.89) / len(values_1_count)) """

        """
            State.
        """
        # 247123161.
        self.assertEqual(state['247123161'][0], 0.82) # speed, phase 0
        self.assertEqual(state['247123161'][1], 3.88) # count, phase 0
        self.assertEqual(state['247123161'][2], 0.74) # speed, phase 1
        self.assertEqual(state['247123161'][3], 2.03) # count, phase 1

        # 247123464.
        self.assertEqual(state['247123464'][0], 0.18) # speed, phase 0
        self.assertEqual(state['247123464'][1], 0.68) # count, phase 0
        self.assertEqual(state['247123464'][2], 0.53) # speed, phase 1
        self.assertEqual(state['247123464'][3], 0.32) # count, phase 1

        # 247123468.
        self.assertEqual(state['247123468'][0], 0.74) # speed, phase 0
        self.assertEqual(state['247123468'][1], 1.27) # count, phase 0
        self.assertEqual(state['247123468'][2], 0.70) # speed, phase 1
        self.assertEqual(state['247123468'][3], 0.55) # count, phase 1

        """
            Reward.
        """
        reward = self.reward(self.observation_space)
        self.assertEqual(reward['247123161'], -0.01*(0.82*3.88 + 0.74*2.03))
        self.assertEqual(reward['247123464'], -0.01*(0.18*0.68 + 0.53*0.32))
        self.assertEqual(reward['247123468'], -0.01*(0.74*1.27 + 0.70*0.55))


    def test_speed_score(self):

        mdp_params = MDPParams(
                        features=('speed_score', 'count'),
                        reward='reward_max_speed_score',
                        normalize_state_space=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None)

        self.observation_space = State(self.network, mdp_params)
        self.observation_space.reset()
        self.reward = build_rewards(mdp_params)
    
        with open('tests/data/grid_kernel_data.dat', "rb") as f:
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

        self.assertEqual(len(state['247123161']), 4)
        self.assertEqual(len(state['247123464']), 4)
        self.assertEqual(len(state['247123468']), 4)

        """ print(state)
        values_0_count = []
        values_1_count = []
        ID = '247123468'
        for t in kernel_data:
            values_0_count.extend(t[ID][0])
            values_1_count.extend(t[ID][1])
        print('count0', len(values_0_count) / 60)
        print('count1', len(values_1_count) / 60)

        vehs_speeds_0 = []
        vehs_speeds_1 = []
        for veh in values_0_count:
            vehs_speeds_0.append(veh.speed)
        for veh in values_1_count:
            vehs_speeds_1.append(veh.speed)

        vehs_speeds_0 = np.array(vehs_speeds_0)
        vehs_speeds_1 = np.array(vehs_speeds_1)

        print(np.sum(vehs_speeds_0) / (13.89*len(values_0_count)))
        print(np.sum(vehs_speeds_1) / (13.89*len(values_1_count))) """

        """
            State.
        """
        # 247123161.
        self.assertEqual(state['247123161'][0], 0.18) # speed score, phase 0
        self.assertEqual(state['247123161'][1], 3.88) # count, phase 0
        self.assertEqual(state['247123161'][2], 0.27) # speed score, phase 1
        self.assertEqual(state['247123161'][3], 2.03) # count, phase 1

        # 247123464.
        self.assertEqual(state['247123464'][0], 0.82) # speed score, phase 0
        self.assertEqual(state['247123464'][1], 0.68) # count, phase 0
        self.assertEqual(state['247123464'][2], 0.47) # speed score, phase 1
        self.assertEqual(state['247123464'][3], 0.32) # count, phase 1

        # 247123468.
        self.assertEqual(state['247123468'][0], 0.27) # speed score, phase 0
        self.assertEqual(state['247123468'][1], 1.27) # count, phase 0
        self.assertEqual(state['247123468'][2], 0.30) # speed score, phase 1
        self.assertEqual(state['247123468'][3], 0.55) # count, phase 1

        """
            Reward.
        """
        reward = self.reward(self.observation_space)
        self.assertEqual(reward['247123161'], 0.01*(0.18*3.88 + 0.27*2.03))
        self.assertEqual(reward['247123464'], 0.01*(0.82*0.68 + 0.47*0.32))
        self.assertEqual(reward['247123468'], 0.01*(0.27*1.27 + 0.30*0.55))


    def test_delay(self):

        mdp_params = MDPParams(
                        features=('delay',),
                        reward='reward_min_delay',
                        normalize_state_space=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)

        self.observation_space = State(self.network, mdp_params)
        self.observation_space.reset()
        self.reward = build_rewards(mdp_params)
    
        with open('tests/data/grid_kernel_data.dat', "rb") as f:
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

        self.assertEqual(len(state['247123161']), 2)
        self.assertEqual(len(state['247123464']), 2)
        self.assertEqual(len(state['247123468']), 2)

        """ print(state)
        values_0_count = []
        values_1_count = []
        ID = '247123468'
        for t in kernel_data:
            values_0_count.extend(t[ID][0])
            values_1_count.extend(t[ID][1])

        vehs_speeds_0 = []
        vehs_speeds_1 = []
        for veh in values_0_count:
            vehs_speeds_0.append(veh.speed)
        for veh in values_1_count:
            vehs_speeds_1.append(veh.speed)

        vehs_speeds_0 = np.array(vehs_speeds_0)
        vehs_speeds_1 = np.array(vehs_speeds_1)

        print(np.sum(np.where(vehs_speeds_0 / 13.89 < 0.1, 1, 0)) / 60)
        print(np.sum(np.where(vehs_speeds_1 / 13.89 < 0.1, 1, 0)) / 60) """

        """
            State.
        """
        # 247123161.
        self.assertEqual(state['247123161'][0], 2.85) # delay, phase 0
        self.assertEqual(state['247123161'][1], 1.18) # delay, phase 1

        # 247123464.
        self.assertEqual(state['247123464'][0], 0.0) # delay, phase 0
        self.assertEqual(state['247123464'][1], 0.08) # delay, phase 1

        # 247123468.
        self.assertEqual(state['247123468'][0], 0.58) # delay, phase 0
        self.assertEqual(state['247123468'][1], 0.27) # delay, phase 1

        """
            Reward.
        """
        reward = self.reward(self.observation_space)
        self.assertEqual(reward['247123161'], -0.01*(2.85 + 1.18))
        self.assertEqual(reward['247123464'], -0.01*(0.0  + 0.08))
        self.assertEqual(reward['247123468'], -0.01*(0.58 + 0.27))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()