import numpy as np

import unittest
from collections import defaultdict

from ilurl.state import State
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.network.test_grid import MAX_VEHS_PER_LANE
from tests.mdp.test_mdp_base import TestGridMDPSetUp


class TestGridQueueCycle1(TestGridMDPSetUp):
    """
        * Tests queue wrt Grid network

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
                        features=('queue', 'lag[queue]'),
                        reward='reward_min_queue_squared',
                        normalize_velocities=True,
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

        super(TestGridQueueCycle1, self).setUp()


    def test_state(self):
        self.assertEqual(len(self.state['247123161']), 4)
        self.assertEqual(len(self.state['247123464']), 4)
        self.assertEqual(len(self.state['247123468']), 4)

    def test_queue_tl1ph0(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 1.0) # queue, phase 0, feature 1
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # queue, phase 0, feature 2

    def test_queue_tl1ph1(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid)

        # 3) Assert 247123161
        self.assertEqual(check_1, 3.0) # queue, phase 1, feature 1
        self.assertEqual(check_1, sol) # queue, phase 1, feature 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl1(self):
        nid ='247123161'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[nid], round(-0.01*(1.0**2 + 3.0**2), 4))

    def test_queue_tl2ph0(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid)

        # 3) Assert 247123464
        self.assertEqual(check_1, 1.0) # queue, phase 0
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_queue_tl2ph1(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid)

        # 3) Assert 247123464
        self.assertEqual(check_1, 1.0) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123464 prevous cycle
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl2(self):
        nid ='247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[nid], round(-0.01*(1.0**2 + 1.0**2), 4))


    def test_queue_tl3ph0(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid)

        # 3) Assert 247123468 actual cycle
        self.assertEqual(check_1, 3.0) # queue, phase 0
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123468 prevous cycle
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_queue_tl3ph1(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid)

        # 3) Assert 247123468
        self.assertEqual(check_1, 2.0) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123468 prevous cycle
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[node_id], round(-0.01*(3.0**2 + 2.0**2), 4))

    def tearDown(self):
        pass


class TestGridQueueCycle1Norm(TestGridQueueCycle1):
    """
        * Tests queue wrt Grid network

        * Extends TestGridQueueCycle1 by normalizing

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
                        features=('queue', 'lag[queue]'),
                        reward='reward_min_queue_squared',
                        normalize_velocities=True,
                        normalize_vehicles=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""

        super(TestGridQueueCycle1Norm, self).setUp()


    def test_queue_tl1ph0(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid,
                            norm_vehs=True)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 0.11) # queue, phase 0, feature 1
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # queue, phase 0, feature 2

    def test_queue_tl1ph1(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]

        sol = process_queue(self.kernel_data_1, nid, pid,
                            norm_vehs=True)

        # 3) Assert 247123161
        self.assertEqual(check_1, 0.38) # queue, phase 1, feature 1
        self.assertEqual(check_1, sol) # queue, phase 1, feature 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl1(self):
        nid ='247123161'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[nid], round(-0.01*(0.11**2 + 0.38**2), 4))

    def test_queue_tl2ph0(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid,
                            norm_vehs=True)

        # 3) Assert 247123464
        self.assertEqual(check_1, 0.12) # queue, phase 0
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_queue_tl2ph1(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid,
                            norm_vehs=True)

        # 3) Assert 247123464
        self.assertEqual(check_1, 0.11) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123464 prevous cycle
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl2(self):
        nid ='247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[nid], round(-0.01*(0.12**2 + 0.11**2), 4))


    def test_queue_tl3ph0(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid,
                            norm_vehs=True)

        # 3) Assert 247123468 actual cycle
        self.assertEqual(check_1,  0.38) # queue, phase 0
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123468 prevous cycle
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_queue_tl3ph1(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid,
                            norm_vehs=True)

        # 3) Assert 247123468
        self.assertEqual(check_1, 0.22) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123468 prevous cycle
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[node_id], round(-0.01*(0.38**2 + 0.22**2), 4))

    def tearDown(self):
        pass

class TestGridQueueCycle2(TestGridQueueCycle1):
    """
        * Tests queue wrt Grid network

        * Extends TestGridQueueCycle1 by running an
         extra cycle

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """

    @lazy_property
    def observation_space(self):
        observation_space = super(TestGridQueueCycle2, self).observation_space
        # Fake environment interaction with state object.
        timesteps = list(range(1,60)) + [0]

        for t, data in zip(timesteps, self.kernel_data_2):
            observation_space.update(t, data)

        return observation_space

    def setUp(self):
        """Code here will run before every test"""

        super(TestGridQueueCycle1, self).setUp()

    def test_state(self):
        self.assertEqual(len(self.state['247123161']), 4)
        self.assertEqual(len(self.state['247123464']), 4)
        self.assertEqual(len(self.state['247123468']), 4)

    def test_queue_tl1ph0(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161 actual cycle
        self.assertEqual(check_1, 1.0) # queue, phase 0, feature 1
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123161 previous cycle
        self.assertEqual(check_2, 1.0) # queue, phase 0, feature 2

    def test_queue_tl1ph1(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161
        self.assertEqual(check_1, 1.0) # queue, phase 1, feature 1
        self.assertEqual(check_1, sol) # queue, phase 1, feature 1

        # 4) Assert 247123161 previous cycle
        self.assertEqual(check_2, 3.0) # queue, phase 1, feature 2

    def test_min_queue_squared_tl1(self):
        nid ='247123161'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(
            reward[nid],
            round(0.01*((1.0**2 + 3.0**2) - (1.0**2 + 1.0**2)), 4)
        )

    def test_queue_tl2ph0(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123464
        self.assertEqual(check_1, 1.0) # queue, phase 0
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123161 previous cycle
        self.assertEqual(check_2, 1.0) # queue, phase 1, feature 2

    def test_queue_tl2ph1(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123464
        self.assertEqual(check_1, 2.0) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123464 previous cycle
        self.assertEqual(check_2, 1.0) # queue, phase 1, feature 2

    def test_min_queue_squared_tl2(self):
        nid ='247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(
            reward[nid],
            round(0.01*((1.0**2 + 1.0**2) - (1.0**2 + 2.0**2)), 4)
        )

    def test_queue_tl3ph0(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123468 actual cycle
        self.assertEqual(check_1, 2.0) # queue, phase 0
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123468 previous cycle
        self.assertEqual(check_2, 3.0) # queue, phase 1, feature 2

    def test_queue_tl3ph1(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123468
        self.assertEqual(check_1, 2.0) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123468 previous cycle
        self.assertEqual(check_2, 2.0) # queue, phase 1, feature 2

    def test_min_queue_squared_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(
            reward[node_id],
            round(0.01*((3.0**2 + 2.0**2) - (2.0**2 + 2.0**2)), 4)
        )

    def tearDown(self):
        pass


def process_queue(data, node_id, phase_id, norm_vehs=False):

        def fn(x):
            if (x / 13.89) < 0.1:
                return 1.0
            else:
                return 0.0

        queues = []
        for t in data:

            qt = defaultdict(lambda : 0)
            for veh in t[node_id][phase_id]:

                key = (veh.edge_id, veh.lane)
                qt[key] += fn(veh.speed)

            if len(qt) == 0:
                queues.append(0.0)
            else:
                if norm_vehs:
                    queues.append(
                        max(v / MAX_VEHS_PER_LANE[k]  for k, v in qt.items()))
                else:
                    queues.append(max(qt.values()))

        ret =  round(max(queues), 2)

        return ret

if __name__ == '__main__':
    unittest.main()
