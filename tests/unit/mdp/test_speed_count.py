import numpy as np
from collections import defaultdict

import unittest

from ilurl.state import State
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp

class TestGridSpeedCountSetUp(TestGridMDPSetUp):
    """
        * Tests speed_count wrt Grid network (reward_max_speed_count)

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
                        features=('speed', 'count'),
                        reward='reward_max_speed_count',
                        normalize_velocities=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridSpeedCountSetUp, self).setUp()

class TestGridTLS1SpeedCountSetUp(TestGridSpeedCountSetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS1SpeedCountSetUp, self).setUp()
        self.ID = '247123161'

class TestGridTLS1SpeedCount(TestGridTLS1SpeedCountSetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS1SpeedCount, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_state_0(self):
        sol = process_speed_count(self.kernel_data, self.ID, self.PHASE_0)

        self.assertEqual(self.STATE_0, [[0.25, 0.87], [0.59, 4.21]]) # phase 0
        self.assertEqual(self.STATE_0, sol) # phase 0

    def test_state_1(self):
        sol = process_speed_count(self.kernel_data, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1,[[0.26, 0.85], [0.84, 1.64]]) # phase 0
        self.assertEqual(self.STATE_1, sol) # phase 0

    def test_reward(self):
        r0 = (0.25 * 0.59 + 0.87 * 4.21)
        r1 = (0.26 * 0.84 + 0.85 * 1.64)
        self.assertAlmostEqual(self.REWARD, round(-0.01 * (r0 + r1), 4))

class TestGridTLS2SpeedCountSetUp(TestGridSpeedCountSetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS2SpeedCountSetUp, self).setUp()
        self.ID = '247123464'

class TestGridTLS2SpeedCount(TestGridTLS2SpeedCountSetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS2SpeedCount, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_state_0(self):
        sol = process_speed_count(self.kernel_data, self.ID, self.PHASE_0)
        self.assertEqual(self.STATE_0, [[0.69, 0.31], [1.45, 0.16]]) # phase 0
        self.assertEqual(self.STATE_0, sol) # phase 0

    def test_state_1(self):
        sol = process_speed_count(self.kernel_data, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1,[[0.8, 0.11], [1.21, 0.64]]) # phase 1
        self.assertEqual(self.STATE_1, sol) # phase 1

    def test_reward(self):
        r0 = (0.69 * 1.45 + 0.31 * 0.16)
        r1 = (0.8 * 1.21 + 0.11 * 0.64)
        self.assertAlmostEqual(self.REWARD, round(-0.01 * (r0 + r1), 4))

class TestGridTLS3SpeedCountSetUp(TestGridSpeedCountSetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS3SpeedCountSetUp, self).setUp()
        self.ID = '247123468'

class TestGridTLS3SpeedCount(TestGridTLS3SpeedCountSetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS3SpeedCount, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_state_0(self):
        sol = process_speed_count(self.kernel_data, self.ID, self.PHASE_0)
        self.assertEqual(self.STATE_0, [[0.4, 0.79], [0.63, 1.64]]) # phase 0
        self.assertEqual(self.STATE_0, sol) # phase 0

    def test_state_1(self):
        sol = process_speed_count(self.kernel_data, self.ID, self.PHASE_1)
        self.assertEqual(self.STATE_1, [[0.33, 0.91], [1.0, 0.92]]) # phase 1
        self.assertEqual(self.STATE_1, sol) # phase 1

    # def test_reward(self):
    #     r0 = (0.4 * 0.63 + 0.79 * 1.64)
    #     r1 = (0.33 * 1.0 + 0.91 * 0.92)
    #     self.assertAlmostEqual(self.REWARD, round(-0.01 * (r0 + r1), 4))

    def tearDown(self):
        pass

def process_speed_count(kernel_data, node_id, phase_id, norm_vehs=False):
    """Processes batched speed_count computation"""
    cycle_time = 60

    def green(x, y):
        if y == 0:
            return x[0].upper() in ('G', 'Y')
        else:
            return x[0].upper() not in ('G', 'Y')

    def ind(x, y):
        g = green(x, y)
        return int(g * 0 + (not g) * 1)

    # speed_count_times = []
    weight = [0.0, 0.0]
    timesteps, vehs, tls = zip(*kernel_data)

    red_speeds = []
    green_speeds = []
    speeds = []
    counts = []
    for t, vehicles, tls in zip(timesteps, vehs, tls):
        vehs = vehicles[node_id][phase_id]
        tl = tls[node_id]
        index = ind(tl, phase_id)
        weight[index] += 1
        for veh in vehs:
            if index == 0:
                green_speeds.append(veh.speed)

            if index == 1:
                red_speeds.append(veh.speed)

    green_speeds = np.array(green_speeds)
    red_speeds = np.array(red_speeds)

    green_count = round(float(len(green_speeds)) / weight[0], 2)
    green_speed = round(np.sum(np.maximum((13.89 - green_speeds), 0) / 13.89) / len(green_speeds), 2)

    red_count = round(float(len(red_speeds)) / weight[1], 2)
    red_speed = round(np.sum(np.maximum((13.89 - red_speeds), 0) / 13.89) / len(red_speeds), 2)
    speeds = [green_speed, red_speed]
    counts = [green_count, red_count]

    return [speeds, counts]

if __name__ == '__main__':
    unittest.main()
