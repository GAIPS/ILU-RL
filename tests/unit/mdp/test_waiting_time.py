import numpy as np

import unittest
from collections import defaultdict

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.utils.aux import flatten
from ilurl.utils.properties import lazy_property

from tests.unit.network.test_grid import MAX_VEHS_PER_LANE
from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp


class TestGridWaitingTimeSetUp(TestGridMDPSetUp):
    """
        * Tests waiting time wrt Grid network

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
                        features=('waiting_time',),
                        reward='reward_min_waiting_time',
                        normalize_velocities=True,
                        normalize_vehicles=False,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    # TODO: Move up inheritance chain
    @lazy_property
    def state(self):
        # Get state.
        state = self.observation_space.feature_map(
            categorize=self.mdp_params.discretize_state_space,
            flatten=False
        )
        return state
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridWaitingTimeSetUp, self).setUp()

class TestGridTLS1WaitingTimeSetUp(TestGridWaitingTimeSetUp):
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

        super(TestGridTLS1WaitingTimeSetUp, self).setUp()
        self.ID = '247123161'

class TestGridTLS1WaitingTime(TestGridTLS1WaitingTimeSetUp):
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

        super(TestGridTLS1WaitingTime, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_waiting_time(self.kernel_data, self.ID, self.PHASE_0)

        # 3) Assert 247123161
        self.assertEqual(self.STATE_0, [0.27, 0.0]) # wait_t, phase 0
        self.assertEqual(self.STATE_0, sol) # wait_t, phase 0

    def test_1(self):
        # 2) Define state & solution
        sol = process_waiting_time(self.kernel_data, self.ID, self.PHASE_1)

        # 3) Assert 247123161
        self.assertEqual(self.STATE_1, [3.97, 2.73]) # wait_t, phase 0
        self.assertEqual(self.STATE_1, sol) # wait_t, phase 0

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (0.27 + 0.0 + 3.97 + 2.73))

class TestGridTLS2WaitingTimeSetUp(TestGridWaitingTimeSetUp):
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

        super(TestGridTLS2WaitingTimeSetUp, self).setUp()
        self.ID = '247123464'

class TestGridTLS2WaitingTime(TestGridTLS2WaitingTimeSetUp):
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

        super(TestGridTLS2WaitingTime, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_waiting_time(self.kernel_data, self.ID, self.PHASE_0)

        # 3) Assert 247123464
        self.assertEqual(self.STATE_0, [0.19, 6.15]) # wait_t, phase 0
        self.assertEqual(self.STATE_0, sol) # wait_t, phase 0

    def test_1(self):
        # 2) Define state & solution
        sol = process_waiting_time(self.kernel_data, self.ID, self.PHASE_1)

        # 3) Assert 247123464
        self.assertEqual(self.STATE_1, [0.09, 0.46]) # wait_t, phase 1
        self.assertEqual(self.STATE_1, sol) # wait_t, phase 1

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (0.19 + 6.15 + 0.09 + 0.46))

class TestGridTLS3WaitingTimeSetUp(TestGridWaitingTimeSetUp):
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

        super(TestGridTLS3WaitingTimeSetUp, self).setUp()
        self.ID = '247123468'

class TestGridTLS3WaitingTime(TestGridTLS3WaitingTimeSetUp):
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

        super(TestGridTLS3WaitingTime, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_waiting_time(self.kernel_data, self.ID, self.PHASE_0)

        # 3) Assert 247123468
        self.assertEqual(self.STATE_0, [4.9, 0.13]) # wait_t, phase 0
        self.assertEqual(self.STATE_0, sol) # wait_t, phase 0

    def test_1(self):
        # 2) Define state & solution
        sol = process_waiting_time(self.kernel_data, self.ID, self.PHASE_1)

        # 3) Assert 247123468
        self.assertEqual(self.STATE_1, [0.6, 0.17]) # wait_t, phase 1
        self.assertEqual(self.STATE_1, sol) # wait_t, phase 1

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (4.9 + 0.13 + 0.6 + 0.17))

def process_waiting_time(kernel_data, node_id, phase_id, norm_vehs=False):
    """Processes batched waiting time computation"""
    cycle_time = 60

    def fn(x):
        if (x / 13.89) < 0.1:
            return 1.0
        else:
            return 0.0

    def green(x, y):
        if y == 0:
            return x[0].upper() in ('G', 'Y')
        else:
            return x[0].upper() not in ('G', 'Y')

    def ind(x, y):
        g = green(x, y)
        return int(g * 0 + (not g) * 1)


    wait_times = []
    weight = [0.0, 0.0]
    timesteps = list(range(1, 60)) + [0]
    vehs, tls = zip(*kernel_data)
    for t, vehicles, tls in zip(timesteps, vehs, tls):

        qt = defaultdict(lambda : [0.0, 0.0])
        vehs = vehicles[node_id][phase_id]
        tl = tls[node_id]
        index = ind(tl, phase_id)
        weight[index] += 1
        for veh in vehs:
            key = (veh.edge_id, veh.lane)
            qt[key][index] += fn(veh.speed)

        if len(qt) == 0:
            wait_times.append([0.0, 0.0])
        else:
            if norm_vehs:
                wait_times.append([
                    sum([v / MAX_VEHS_PER_LANE[k]  for k, v in qt.items()]) for i in range(2)
                ])
            else:
                wait_times.append([sum([values[i] for values in qt.values()]) for i in range(2)])

    ret =  [round(sum([wt[i] for wt in wait_times]) / weight[i] , 2) for i in range(2)]
    return ret

if __name__ == '__main__':
    unittest.main()
