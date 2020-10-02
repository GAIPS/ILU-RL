from collections import defaultdict
import unittest

import numpy as np

from ilurl.state import State
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp

class TestGridDelaySetUp(TestGridMDPSetUp):
    """
        * Tests delay wrt Grid network (reward_min_delay)

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

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridDelaySetUp, self).setUp()

class TestGridTLS1DelaySetUp(TestGridDelaySetUp):
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

        super(TestGridTLS1DelaySetUp, self).setUp()
        self.ID = '247123161'

class TestGridTLS1Delay(TestGridTLS1DelaySetUp):
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

        super(TestGridTLS1Delay, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_0)
        # 3) Assert 247123161
        self.assertEqual(self.STATE_0, [0.02, 3.2]) # phase 0
        self.assertEqual(self.STATE_0, sol) # phase 0

    def test_1(self):
        # 2) Define state & solution
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_1)

        # 3) Assert 247123161
        self.assertEqual(self.STATE_1, [0.06, 1.22]) # phase 0
        self.assertEqual(self.STATE_1, sol) # phase 0

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (0.02 + 3.2 + 0.06 + 1.22))

class TestGridTLS2DelaySetUp(TestGridDelaySetUp):
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

        super(TestGridTLS2DelaySetUp, self).setUp()
        self.ID = '247123464'

class TestGridTLS2Delay(TestGridTLS2DelaySetUp):
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

        super(TestGridTLS2Delay, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_0)

        # 3) Assert 247123464
        self.assertEqual(self.STATE_0, [0.75, 0.01]) # wait_t, phase 0
        self.assertEqual(self.STATE_0, sol) # wait_t, phase 0

    def test_1(self):
        # 2) Define state & solution
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_1)

        # 3) Assert 247123464
        self.assertEqual(self.STATE_1, [0.81, 0.01]) # wait_t, phase 1
        self.assertEqual(self.STATE_1, sol) # wait_t, phase 1

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (0.75 + 0.01 + 0.81 + 0.01))

class TestGridTLS3DelaySetUp(TestGridDelaySetUp):
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

        super(TestGridTLS3DelaySetUp, self).setUp()
        self.ID = '247123468'

class TestGridTLS3Delay(TestGridTLS3DelaySetUp):
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

        super(TestGridTLS3Delay, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_0)

        # 3) Assert 247123468
        self.assertEqual(self.STATE_0, [0.09, 1.08]) # wait_t, phase 0
        self.assertEqual(self.STATE_0, sol) # wait_t, phase 0

    def test_1(self):
        # 2) Define state & solution
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_1)

        # 3) Assert 247123468
        self.assertEqual(self.STATE_1, [0.13, 0.74]) # wait_t, phase 1
        self.assertEqual(self.STATE_1, sol) # wait_t, phase 1

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (0.09 + 1.08 + 0.13 + 0.74))

    def tearDown(self):
        pass


# clas# s TestGridMaxDelayReduction1(TestGridMDPSetUp):
#     # """
#     #     * Tests delay wrt Grid network (reward_max_delay_reduction)
#     #     * (part 1)
# 
#     #     * Set of tests that target the implemented
#     #       problem formulations, i.e. state and reward
#     #       function definitions.
# 
#     #     * Use lazy_properties to compute once and use
#     #       as many times as you want -- it's a cached
#     #       property
#     # """
#     # @lazy_property
#     # def mdp_params(self):
#     #     mdp_params = MDPParams(
#     #                     features=('delay', 'lag[delay]'),
#     #                     reward='reward_max_delay_reduction',
#     #                     normalize_velocities=True,
#     #                     normalize_vehicles=False,
#     #                     discretize_state_space=False,
#     #                     reward_rescale=0.01,
#     #                     time_period=None,
#     #                     velocity_threshold=0.1)
#     #     return mdp_params
# 
#     # @lazy_property
#     # def observation_space(self):
#     #     observation_space = State(self.network, self.mdp_params)
#     #     observation_space.reset()
#     #     # Fake environment interaction with state object.
#     #     timesteps = list(range(1,60)) + [0]
# 
#     #     for t, data in zip(timesteps, self.kernel_data_1):
#     #         observation_space.update(t, data)
# 
#     #     return observation_space
# 
# 
#     # def setUp(self):
#     #     """Code here will run before every test"""
#     #     super(TestGridMaxDelayReduction1, self).setUp()
# 
#     # def test_delay_tl1ph0(self):
#     #     # 1) Define constraints
#     #     nid ='247123161'
#     #     pid = 0
#     #     fid = slice(pid, pid + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_1, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.85) # delay, phase 0, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 0
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.00) # delay, phase 0, prev cycle
# 
#     # def test_delay_tl1ph1(self):
#     #     # 1) Define constraints
#     #     nid ='247123161'
#     #     pid = 1
#     #     fid = slice(pid * 2, pid * 2 + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_1, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.82) # delay, phase 1, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 1
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.00) # delay, phase 1, prev cycle
# 
#     # def test_delay_tl2ph0(self):
#     #     # 1) Define constraints
#     #     nid ='247123464'
#     #     pid = 0
#     #     fid = slice(pid, pid + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_1, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.08) # delay, phase 0, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 0
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.00) # delay, phase 0, prev cycle
# 
#     # def test_delay_tl2ph1(self):
#     #     # 1) Define constraints
#     #     nid ='247123464'
#     #     pid = 1
#     #     fid = slice(pid * 2, pid * 2 + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_1, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.16) # delay, phase 1, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 1
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.00) # delay, phase 1, prev cycle
# 
#     # def test_delay_tl3ph0(self):
#     #     # 1) Define constraints
#     #     nid ='247123468'
#     #     pid = 0
#     #     fid = slice(pid, pid + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_1, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 3.73) # delay, phase 0, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 0
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.00) # delay, phase 0, prev cycle
# 
#     # def test_delay_tl3ph1(self):
#     #     # 1) Define constraints
#     #     nid ='247123468'
#     #     pid = 1
#     #     fid = slice(pid * 2, pid * 2 + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_1, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.5) # delay, phase 1, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 1
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.00) # delay, phase 1, prev cycle
# 
#     # def tearDown(self):
#     #     pass
# 
# class TestGridMaxDelayReduction2(TestGridMaxDelayReduction1):
#     # """
#     #     * Tests delay wrt Grid network (reward_max_delay_reduction)
#     #     * (part 2)
# 
#     #     * Set of tests that target the implemented
#     #       problem formulations, i.e. state and reward
#     #       function definitions.
# 
#     #     * Use lazy_properties to compute once and use
#     #       as many times as you want -- it's a cached
#     #       property
#     # """
# 
#     # @lazy_property
#     # def observation_space(self):
#     #     observation_space = super(TestGridMaxDelayReduction2, self).observation_space
#     #     # Fake environment interaction with state object.
#     #     timesteps = list(range(1,60)) + [0]
# 
#     #     for t, data in zip(timesteps, self.kernel_data_2):
#     #         observation_space.update(t, data)
# 
#     #     return observation_space
# 
#     # def setUp(self):
#     #     """Code here will run before every test"""
#     #     super(TestGridMaxDelayReduction2, self).setUp()
# 
#     # def test_state(self):
#     #     self.assertEqual(len(self.state['247123161']), 4)
#     #     self.assertEqual(len(self.state['247123464']), 4)
#     #     self.assertEqual(len(self.state['247123468']), 4)
# 
#     # def test_delay_tl1ph0(self):
#     #     # 1) Define constraints
#     #     nid ='247123161'
#     #     pid = 0
#     #     fid = slice(pid, pid + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_2, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.32) # delay, phase 0, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 0
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.85) # delay, phase 0, prev cycle
# 
#     # def test_delay_tl1ph1(self):
#     #     # 1) Define constraints
#     #     nid ='247123161'
#     #     pid = 1
#     #     fid = slice(pid * 2, pid * 2 + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_2, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.80) # delay, phase 1, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 1
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.82) # delay, phase 1, prev cycle
# 
#     # def test_max_delay_reduction_tl1(self):
#     #     node_id ='247123161'
#     #     reward = self.reward(self.observation_space)
#     #     self.assertAlmostEqual(reward[node_id], round(0.01*(0.85-0.32 + 0.82-0.8), 4))
#     # 
#     # def test_delay_tl2ph0(self):
#     #     # 1) Define constraints
#     #     nid ='247123464'
#     #     pid = 0
#     #     fid = slice(pid, pid + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_2, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 1.52) # delay, phase 0, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 0
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.08) # delay, phase 0, prev cycle
# 
#     # def test_delay_tl2ph1(self):
#     #     # 1) Define constraints
#     #     nid ='247123464'
#     #     pid = 1
#     #     fid = slice(pid * 2, pid * 2 + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_2, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.83) # delay, phase 1, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 1
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.16) # delay, phase 1, prev cycle
# 
#     # def test_max_delay_reduction_tl2(self):
#     #     node_id ='247123464'
#     #     reward = self.reward(self.observation_space)
#     #     self.assertAlmostEqual(reward[node_id], round(0.01*(0.08-1.52 + 0.16-0.83), 4))
# 
#     # def test_delay_tl3ph0(self):
#     #     # 1) Define constraints
#     #     nid ='247123468'
#     #     pid = 0
#     #     fid = slice(pid, pid + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_2, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 2.13) # delay, phase 0, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 0
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 3.73) # delay, phase 0, prev cycle
# 
#     # def test_delay_tl3ph1(self):
#     #     # 1) Define constraints
#     #     nid ='247123468'
#     #     pid = 1
#     #     fid = slice(pid * 2, pid * 2 + 2)
# 
#     #     # 2) Define state & solution
#     #     check_1, check_2 = self.state[nid][fid]
#     #     sol = process_delay(self.kernel_data_2, nid, pid)
# 
#     #     # 3) Assert 247123161 actual cycle
#     #     self.assertEqual(check_1, 0.57) # delay, phase 1, actual cycle
#     #     self.assertEqual(check_1, sol) # delay, phase 1
# 
#     #     # 4) Assert 247123161 previous cycle (no data)
#     #     self.assertEqual(check_2, 0.5) # delay, phase 1, prev cycle
# 
#     # def test_max_delay_reduction_tl3(self):
#     #     node_id ='247123468'
#     #     reward = self.reward(self.observation_space)
#     #     self.assertAlmostEqual(reward[node_id], round(0.01*(3.73-2.13 + 0.50-0.57), 4))

    def tearDown(self):
        pass


def process_delay(kernel_data, node_id, phase_id, norm_vehs=False):
    """Processes batched delay computation"""
    cycle_time = 60

    def fn(x):
        return np.where(x >= 1, 0.0, np.exp(-5 * x))

    def green(x, y):
        if y == 0:
            return x[0].upper() in ('G', 'Y')
        else:
            return x[0].upper() not in ('G', 'Y')

    def ind(x, y):
        g = green(x, y)
        return int(g * 0 + (not g) * 1)


    delay_times = []
    weight = [0.0, 0.0]
    # timesteps = list(range(1, 60)) + [0]
    timesteps, vehs, tls = zip(*kernel_data)
    for t, vehicles, tls in zip(timesteps, vehs, tls):
        qt = defaultdict(lambda : [0.0, 0.0])
        vehs = vehicles[node_id][phase_id]
        tl = tls[node_id]
        index = ind(tl, phase_id)
        weight[index] += 1
        for veh in vehs:
            key = (veh.edge_id, veh.lane)
            qt[key][index] += fn(veh.speed / 13.89)

        if len(qt) == 0:
            delay_times.append([0.0, 0.0])
        else:
            if norm_vehs:
                delay_times.append([
                    sum([v / MAX_VEHS_PER_LANE[k]  for k, v in qt.items()]) for i in range(2)
                ])
            else:
                delay_times.append([sum([values[i] for values in qt.values()]) for i in range(2)])

    ret =  [round(sum([dt[i] for dt in delay_times]) / weight[i] , 2) for i in range(2)]
    return ret

# def process_delay(kernel_data, node_id, phase_id):
#     """Processes batched delay computation"""
#     cycle_time = 60
# 
#     def delay(x):
#             return np.where(x >= 1, 0.0, np.exp(-5*x))
# 
#     values_count = []
#     for t in kernel_data:
#         values_count.extend(t[node_id][phase_id])
# 
#     vehs_speeds = []
#     for veh in values_count:
#         vehs_speeds.append(veh.speed)
# 
#     vehs_speeds = np.array(vehs_speeds)
# 
#     ret = np.sum(delay(vehs_speeds / 13.89)) / cycle_time
#     ret = round(ret, 2)
#     return ret

if __name__ == '__main__':
    unittest.main()
