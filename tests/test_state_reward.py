import pickle
import unittest

import numpy as np

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
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


    def test_max_flow(self):
        """
            Maximize flow.
        """
        mdp_params = MDPParams(
                        features=('flow',),
                        reward='reward_max_flow',
                        normalize_velocities=True,
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

        # Assert that number of phases == 2
        self.assertEqual(len(state['247123161']), 2)
        self.assertEqual(len(state['247123464']), 2)
        self.assertEqual(len(state['247123468']), 2)


        """print(state)
        ID = '247123161'
        veh_0_set = set()
        veh_1_set = set()
        for t, data in zip(timesteps, kernel_data):
            if t == 0:
                veh_0_0_set =  {veh.id for veh in data[ID][0]}
                veh_0_1_set =  {veh.id for veh in data[ID][1]}
            else:
                veh_0_set = veh_0_set.union({veh.id for veh in data[ID][0]})
                veh_1_set = veh_1_set.union({veh.id for veh in data[ID][1]})

        print(len(veh_0_set - veh_0_0_set))
        print(len(veh_1_set - veh_0_1_set))"""

        # State.
        # 247123161.
        self.assertEqual(state['247123161'][0], 7.0) # flow, phase 0
        self.assertEqual(state['247123161'][1], 9.0) # flow, phase 1

        # 247123464.
        self.assertEqual(state['247123464'][0], 9) # flow, phase 0
        self.assertEqual(state['247123464'][1], 1) # flow, phase 1

        # 247123468.
        self.assertEqual(state['247123468'][0], 15) # flow, phase 0
        self.assertEqual(state['247123468'][1], 2) # flow, phase 1

        # Reward.
        reward = self.reward(self.observation_space)
        self.assertEqual(reward['247123161'], 0.01*(7.0 + 9.0))
        self.assertEqual(reward['247123464'], 0.01*(9.0  + 1.0))
        self.assertEqual(reward['247123468'], 0.01*(15 + 2))

    def test_max_speed_count(self):
        """
            Maximize weighted average speed.
        """
        mdp_params = MDPParams(
                        features=('speed', 'count'),
                        reward='reward_max_speed_count',
                        normalize_velocities=True,
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

        # State.
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

        # Reward.
        reward = self.reward(self.observation_space)
        self.assertEqual(reward['247123161'], -0.01*(0.82*3.88 + 0.74*2.03))
        self.assertEqual(reward['247123464'], -0.01*(0.18*0.68 + 0.53*0.32))
        self.assertEqual(reward['247123468'], -0.01*(0.74*1.27 + 0.70*0.55))


    def test_max_speed_score(self):
        """
            Maximize weighted average speed score.
        """
        mdp_params = MDPParams(
                        features=('speed_score', 'count'),
                        reward='reward_max_speed_score',
                        normalize_velocities=True,
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

        # State.
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

        # Reward.
        reward = self.reward(self.observation_space)
        self.assertEqual(reward['247123161'], 0.01*(0.18*3.88 + 0.27*2.03))
        self.assertEqual(reward['247123464'], 0.01*(0.82*0.68 + 0.47*0.32))
        self.assertEqual(reward['247123468'], 0.01*(0.27*1.27 + 0.30*0.55))


    def test_min_delay(self):
        """
            Minimize delay.
        """
        mdp_params = MDPParams(
                        features=('delay',),
                        reward='reward_min_delay',
                        normalize_velocities=True,
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

        """
        # OLD = WAITING TIME
        print(state)
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

        """ def delay(x):
            return np.where(x >= 1, 0.0, np.exp(-5*x))

        print(state)
        values_0_count = []
        values_1_count = []
        ID = '247123464'
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

        print(np.sum(delay(vehs_speeds_0 / 13.89)) / 60)
        print(np.sum(delay(vehs_speeds_1 / 13.89)) / 60) """

        # State.
        # 247123161.
        self.assertEqual(state['247123161'][0], 2.91) # delay, phase 0
        self.assertEqual(state['247123161'][1], 1.23) # delay, phase 1

        # 247123464.
        self.assertEqual(state['247123464'][0], 0.02) # delay, phase 0
        self.assertEqual(state['247123464'][1], 0.09) # delay, phase 1

        # 247123468.
        self.assertEqual(state['247123468'][0], 0.69) # delay, phase 0
        self.assertEqual(state['247123468'][1], 0.29) # delay, phase 1

        # Reward.
        reward = self.reward(self.observation_space)
        self.assertEqual(reward['247123161'], -0.01*(2.91 + 1.23))
        self.assertEqual(reward['247123464'], -0.01*(0.02 + 0.09))
        self.assertEqual(reward['247123468'], -0.01*(0.69 + 0.29))


    def test_max_delay_reduction(self):
        """
            Maximize delay reduction.
        """
        mdp_params = MDPParams(
                        features=('delay', 'lag[delay]'),
                        reward='reward_max_delay_reduction',
                        normalize_velocities=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)

        self.observation_space = State(self.network, mdp_params)
        self.observation_space.reset()
        self.reward = build_rewards(mdp_params)
    
        with open('tests/data/grid_kernel_data_1.dat', "rb") as f:
            kernel_data_1 = pickle.load(f)

        self.assertEqual(len(kernel_data_1), 60)

        with open('tests/data/grid_kernel_data_2.dat', "rb") as f:
            kernel_data_2 = pickle.load(f)

        self.assertEqual(len(kernel_data_2), 60)

        # Fake environment interaction with state object (cycle 1).
        timesteps = list(range(1,60)) + [0]
        for t, data in zip(timesteps, kernel_data_1):
            self.observation_space.update(t, data)

        # Get state.
        state = self.observation_space.feature_map(
            categorize=mdp_params.discretize_state_space,
            flatten=True
        )

        self.assertEqual(len(state['247123161']), 4)
        self.assertEqual(len(state['247123464']), 4)
        self.assertEqual(len(state['247123468']), 4)

        def delay(x):
            return np.where(x >= 1, 0.0, np.exp(-5*x))

        """ print(state)
        for t_id in ['247123161', '247123464' ,'247123468']:
            values_0_count = []
            values_1_count = []

            for t in kernel_data_1:
                values_0_count.extend(t[t_id][0])
                values_1_count.extend(t[t_id][1])

            vehs_speeds_0 = []
            vehs_speeds_1 = []
            for veh in values_0_count:
                vehs_speeds_0.append(veh.speed)
            for veh in values_1_count:
                vehs_speeds_1.append(veh.speed)

            vehs_speeds_0 = np.array(vehs_speeds_0)
            vehs_speeds_1 = np.array(vehs_speeds_1)

            print('t_id={0}, delay (phase 0):'.format(t_id), np.sum(delay(vehs_speeds_0 / 13.89)) / 60)
            print('t_id={0}, delay (phase 1):'.format(t_id), np.sum(delay(vehs_speeds_1 / 13.89)) / 60) """
        
        # State.
        # 247123161.
        self.assertEqual(state['247123161'][0], 0.85) # delay, phase 0, actual cycle
        self.assertEqual(state['247123161'][1], 0.0)  # delay, phase 0, previous cycle (No data yet)
        self.assertEqual(state['247123161'][2], 0.82) # delay, phase 1, actual cycle
        self.assertEqual(state['247123161'][3], 0.0)  # delay, phase 1, previous cycle (No data yet)

        # 247123464.
        self.assertEqual(state['247123464'][0], 0.08) # delay, phase 0, actual cycle
        self.assertEqual(state['247123464'][1], 0.0)  # delay, phase 0, previous cycle (No data yet)
        self.assertEqual(state['247123464'][2], 0.16) # delay, phase 1, actual cycle
        self.assertEqual(state['247123464'][3], 0.0)  # delay, phase 1, previous cycle (No data yet)

        # 247123468.
        self.assertEqual(state['247123468'][0], 3.73) # delay, phase 0, actual cycle
        self.assertEqual(state['247123468'][1], 0.0)  # delay, phase 0, previous cycle (No data yet)
        self.assertEqual(state['247123468'][2], 0.50) # delay, phase 1, actual cycle
        self.assertEqual(state['247123468'][3], 0.0)  # delay, phase 1, previous cycle (No data yet)

        # Fake environment interaction with state object (cycle 2).
        timesteps = list(range(1,60)) + [0]
        for t, data in zip(timesteps, kernel_data_2):
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
        for t_id in ['247123161', '247123464' ,'247123468']:
            values_0_count = []
            values_1_count = []

            for t in kernel_data_2:
                values_0_count.extend(t[t_id][0])
                values_1_count.extend(t[t_id][1])

            vehs_speeds_0 = []
            vehs_speeds_1 = []
            for veh in values_0_count:
                vehs_speeds_0.append(veh.speed)
            for veh in values_1_count:
                vehs_speeds_1.append(veh.speed)

            vehs_speeds_0 = np.array(vehs_speeds_0)
            vehs_speeds_1 = np.array(vehs_speeds_1)

            print('t_id={0}, delay (phase 0):'.format(t_id), np.sum(delay(vehs_speeds_0 / 13.89)) / 60)
            print('t_id={0}, delay (phase 1):'.format(t_id), np.sum(delay(vehs_speeds_1 / 13.89)) / 60) """

        # State.
        # 247123161.
        self.assertEqual(state['247123161'][0], 0.32) # delay, phase 0, actual cycle
        self.assertEqual(state['247123161'][1], 0.85) # delay, phase 0, previous cycle
        self.assertEqual(state['247123161'][2], 0.80) # delay, phase 1, actual cycle
        self.assertEqual(state['247123161'][3], 0.82) # delay, phase 1, previous cycle

        # 247123464.
        self.assertEqual(state['247123464'][0], 1.52) # delay, phase 0, actual cycle
        self.assertEqual(state['247123464'][1], 0.08) # delay, phase 0, previous cycle
        self.assertEqual(state['247123464'][2], 0.83) # delay, phase 1, actual cycle
        self.assertEqual(state['247123464'][3], 0.16) # delay, phase 1, previous cycle

        # 247123468.
        self.assertEqual(state['247123468'][0], 2.13) # delay, phase 0, actual cycle
        self.assertEqual(state['247123468'][1], 3.73) # delay, phase 0, previous cycle
        self.assertEqual(state['247123468'][2], 0.57) # delay, phase 1, actual cycle
        self.assertEqual(state['247123468'][3], 0.50) # delay, phase 1, previous cycle

        # Reward.
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward['247123161'], round(0.01*(0.85-0.32 + 0.82-0.80), 4))
        self.assertAlmostEqual(reward['247123464'], round(0.01*(0.08-1.52 + 0.16-0.83), 4))
        self.assertAlmostEqual(reward['247123468'], round(0.01*(3.73-2.13 + 0.50-0.57), 4))




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

        self.assertEqual(len(state['247123161']), 3)
        self.assertEqual(len(state['247123464']), 3)
        self.assertEqual(len(state['247123468']), 3)

        # State.
        # 247123161.
        self.assertEqual(state['247123161'][0], 0)    # time variable
        self.assertEqual(state['247123161'][1], 2.91) # delay, phase 0
        self.assertEqual(state['247123161'][2], 1.23) # delay, phase 1

        # 247123464.
        self.assertEqual(state['247123464'][0], 0)    # time variable
        self.assertEqual(state['247123464'][1], 0.02)  # delay, phase 0
        self.assertEqual(state['247123464'][2], 0.09) # delay, phase 1

        # 247123468.
        self.assertEqual(state['247123468'][0], 0)    # time variable
        self.assertEqual(state['247123468'][1], 0.69) # delay, phase 0
        self.assertEqual(state['247123468'][2], 0.29) # delay, phase 1

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
