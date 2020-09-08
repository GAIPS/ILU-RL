import numpy as np

import unittest

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.utils.aux import flatten
from ilurl.utils.properties import lazy_property

from tests.network.test_grid import TestGridBase, MAX_VEHS


class TestGridDelay(TestGridBase):
    """
        * Tests delay wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    @lazy_property
    def mdp_params(self):
        mdp_params = MDPParams(
                        features=('delay',),
                        reward='reward_min_delay',
                        normalize_velocities=True,
                        normalize_vehicles=False,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    @lazy_property
    def observation_space(self):
        observation_space = State(self.network, self.mdp_params)
        observation_space.reset()
        # Fake environment interaction with state object.
        timesteps = list(range(1,60)) + [0]

        for t, data in zip(timesteps, self.kernel_data):
            observation_space.update(t, data)

        return observation_space

    @lazy_property
    def reward(self):
        reward = build_rewards(self.mdp_params)
        return reward

    @lazy_property
    def state(self):
        # Get state.
        state = self.observation_space.feature_map(
            categorize=self.mdp_params.discretize_state_space,
            flatten=True
        )
        return state

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridDelay, self).setUp()


    def test_delay_tl1ph0(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check, 2.91) # delay, phase 0
        self.assertEqual(check, sol) # delay, phase 0

    def test_delay_tl1ph1(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check, 1.23) # delay, phase 1
        self.assertEqual(check, sol) # delay, phase 1

    def test_min_delay_tl1(self):
        node_id ='247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], -0.01*(2.91 + 1.23))

    def test_delay_tl2ph0(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check, 0.02) # delay, phase 0
        self.assertEqual(check, sol) # delay, phase 0

    def test_delay_tl2ph1(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check, 0.09) # delay, phase 1
        self.assertEqual(check, sol) # delay, phase 1

    def test_min_delay_tl2(self):
        node_id ='247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], -0.01*(0.02 + 0.09))


    def test_delay_tl3ph0(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check,  0.69) # delay, phase 0
        self.assertEqual(check, sol) # delay, phase 0

    def test_delay_tl3ph1(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check, 0.29) # delay, phase 1
        self.assertEqual(check, sol) # delay, phase 1

    def test_min_delay_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], -0.01*(0.69 + 0.29))

    def tearDown(self):
        pass


def process_delay(kernel_data, node_id, phase_id):
    """Processes batched delay computation"""

    def delay(x):
            return np.where(x >= 1, 0.0, np.exp(-5*x))

    values_count = []
    for t in kernel_data:
        values_count.extend(t[node_id][phase_id])

    vehs_speeds = []
    for veh in values_count:
        vehs_speeds.append(veh.speed)

    vehs_speeds = np.array(vehs_speeds)

    ret = np.sum(delay(vehs_speeds / 13.89)) / 60
    ret = round(ret, 2)
    return ret

if __name__ == '__main__':
    unittest.main()
