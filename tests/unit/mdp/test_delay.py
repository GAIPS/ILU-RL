import numpy as np

import unittest

from ilurl.state.state import State
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp

class TestGridMinDelay(TestGridMDPSetUp):
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
        super(TestGridMinDelay, self).setUp()

    def test_delay_tl1ph0(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check, 1.35) # delay, phase 0
        self.assertEqual(check, sol) # delay, phase 0

    def test_delay_tl1ph1(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check, 1.38) # delay, phase 1
        self.assertEqual(check, sol) # delay, phase 1

    def test_min_delay_tl1(self):
        node_id ='247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(1.35 + 1.38), 4))

    def test_delay_tl2ph0(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check, 0.01) # delay, phase 0
        self.assertEqual(check, sol) # delay, phase 0

    def test_delay_tl2ph1(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check, 0.62) # delay, phase 1
        self.assertEqual(check, sol) # delay, phase 1

    def test_min_delay_tl2(self):
        node_id ='247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(0.01 + 0.62), 4))


    def test_delay_tl3ph0(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check, 0.85) # delay, phase 0
        self.assertEqual(check, sol) # delay, phase 0

    def test_delay_tl3ph1(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_delay(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check, 0.9) # delay, phase 1
        self.assertEqual(check, sol) # delay, phase 1

    def test_min_delay_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(0.85 + 0.9), 4))

    def tearDown(self):
        pass


class TestGridMaxDelayReduction1(TestGridMDPSetUp):
    """
        * Tests delay wrt Grid network (reward_max_delay_reduction)
        * (part 1)

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
                        features=('delay', 'lag[delay]'),
                        reward='reward_max_delay_reduction',
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

        for t, data in zip(timesteps, self.kernel_data_1):
            observation_space.update(t, data)

        return observation_space


    def setUp(self):
        """Code here will run before every test"""
        super(TestGridMaxDelayReduction1, self).setUp()

    def test_delay_tl1ph0(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_1, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 1.35) # delay, phase 0, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # delay, phase 0, prev cycle

    def test_delay_tl1ph1(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_1, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 1.38) # delay, phase 1, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # delay, phase 1, prev cycle

    def test_delay_tl2ph0(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_1, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.01) # delay, phase 0, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # delay, phase 0, prev cycle

    def test_delay_tl2ph1(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_1, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.62) # delay, phase 1, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # delay, phase 1, prev cycle

    def test_delay_tl3ph0(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_1, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.85) # delay, phase 0, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # delay, phase 0, prev cycle

    def test_delay_tl3ph1(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_1, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.9) # delay, phase 1, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # delay, phase 1, prev cycle

    def tearDown(self):
        pass

class TestGridMaxDelayReduction2(TestGridMaxDelayReduction1):
    """
        * Tests delay wrt Grid network (reward_max_delay_reduction)
        * (part 2)

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """

    @lazy_property
    def observation_space(self):
        observation_space = super(TestGridMaxDelayReduction2, self).observation_space
        # Fake environment interaction with state object.
        timesteps = list(range(1,60)) + [0]

        for t, data in zip(timesteps, self.kernel_data_2):
            observation_space.update(t, data)

        return observation_space

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridMaxDelayReduction2, self).setUp()

    def test_state(self):
        self.assertEqual(len(self.state['247123161']), 4)
        self.assertEqual(len(self.state['247123464']), 4)
        self.assertEqual(len(self.state['247123468']), 4)

    def test_delay_tl1ph0(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.1) # delay, phase 0, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 1.35) # delay, phase 0, prev cycle

    def test_delay_tl1ph1(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.92) # delay, phase 1, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 1.38) # delay, phase 1, prev cycle

    def test_max_delay_reduction_tl1(self):
        node_id ='247123161'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[node_id], round(0.01*(1.35-0.1 + 1.38-0.92), 4))
    
    def test_delay_tl2ph0(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.64) # delay, phase 0, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.01) # delay, phase 0, prev cycle

    def test_delay_tl2ph1(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 1.46) # delay, phase 1, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.62) # delay, phase 1, prev cycle

    def test_max_delay_reduction_tl2(self):
        node_id ='247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[node_id], round(0.01*(0.01-0.64 + 0.62-1.46), 4))

    def test_delay_tl3ph0(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.26) # delay, phase 0, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.85) # delay, phase 0, prev cycle

    def test_delay_tl3ph1(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_delay(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 2.64) # delay, phase 1, actual cycle
        self.assertEqual(check_1, sol) # delay, phase 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.9) # delay, phase 1, prev cycle

    def test_max_delay_reduction_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[node_id], round(0.01*(0.85-0.26 + 0.9-2.64), 4))

    def tearDown(self):
        pass


def process_delay(kernel_data, node_id, phase_id):
    """Processes batched delay computation"""
    cycle_time = 60

    def delay(x):
            return np.where(x >= 1, 0.0, np.exp(-5*x))

    values_count = []
    for t in kernel_data:
        values_count.extend(t[node_id][phase_id])

    vehs_speeds = []
    for veh in values_count:
        vehs_speeds.append(veh.speed)

    vehs_speeds = np.array(vehs_speeds)

    ret = np.sum(delay(vehs_speeds / 13.89)) / cycle_time
    ret = round(ret, 2)
    return ret

if __name__ == '__main__':
    unittest.main()
