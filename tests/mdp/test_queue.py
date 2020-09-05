import numpy as np

import unittest
# from copy import deepcopy

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.utils.aux import flatten
from ilurl.utils.properties import lazy_property

from tests.network.test_grid import TestGridBase, MAX_VEHS


class TestGridQueueCycle1(TestGridBase):
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

    @lazy_property
    def state(self):
        # Get state.
        state = self.observation_space.feature_map(
            categorize=self.mdp_params.discretize_state_space,
            flatten=True
        )
        return state

    @lazy_property
    def reward(self):
        reward = build_rewards(self.mdp_params)
        return reward


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
        self.assertEqual(check_1, 0.63) # queue, phase 0, feature 1
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
        self.assertEqual(check_1, 0.58) # queue, phase 1, feature 1
        self.assertEqual(check_1, sol) # queue, phase 1, feature 1

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl1(self):
        nid ='247123161'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[nid], -0.01*(0.58**2 + 0.63**2))

    def test_queue_tl2ph0(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid)

        # 3) Assert 247123464
        self.assertEqual(check_1, 0.07) # queue, phase 0
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
        self.assertEqual(check_1, 0.15) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123464 prevous cycle
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl2(self):
        nid ='247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[nid], -0.01*(0.07**2 + 0.15**2))


    def test_queue_tl3ph0(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_1, nid, pid)

        # 3) Assert 247123468 actual cycle
        self.assertEqual(check_1,  2.05) # queue, phase 0
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
        self.assertEqual(check_1, 0.45) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123468 prevous cycle
        self.assertEqual(check_2, 0.00) # queue, phase 1, feature 2

    def test_min_queue_squared_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[node_id], -0.01*(2.05**2 + 0.45**2))

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

    @lazy_property
    def reward(self):
        reward = build_rewards(self.mdp_params)
        return reward


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
        self.assertEqual(check_1, 0.30) # queue, phase 0, feature 1
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123161 previous cycle (no data)
        self.assertEqual(check_2, 0.63) # queue, phase 0, feature 2

    def test_queue_tl1ph1(self):
        # 1) Define constraints
        nid ='247123161'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123161
        self.assertEqual(check_1, 0.48) # queue, phase 1, feature 1
        self.assertEqual(check_1, sol) # queue, phase 1, feature 1

        # 4) Assert 247123161 previous cycle
        self.assertEqual(check_2, 0.58) # queue, phase 1, feature 2

    def test_min_queue_squared_tl1(self):
        nid ='247123161'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(
            reward[nid],
            -0.01*((0.30**2 + 0.48**2) - (0.58**2 + 0.63**2))
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
        self.assertEqual(check_1, 0.63) # queue, phase 0
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123161 previous cycle
        self.assertEqual(check_2, 0.07) # queue, phase 1, feature 2

    def test_queue_tl2ph1(self):
        # 1) Define constraints
        nid ='247123464'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123464
        self.assertEqual(check_1, 0.80) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123464 prevous cycle
        self.assertEqual(check_2, 0.15) # queue, phase 1, feature 2

    def test_min_queue_squared_tl2(self):
        nid ='247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(
            reward[nid],
            -0.01*((0.63**2 + 0.8**2) - (0.07**2 + 0.15**2)))


    def test_queue_tl3ph0(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 0
        fid = slice(pid, pid + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123468 actual cycle
        self.assertEqual(check_1,  1.05) # queue, phase 0
        self.assertEqual(check_1, sol) # queue, phase 0

        # 4) Assert 247123468 prevous cycle
        self.assertEqual(check_2, 2.05) # queue, phase 1, feature 2

    def test_queue_tl3ph1(self):
        # 1) Define constraints
        nid ='247123468'
        pid = 1
        fid = slice(pid * 2, pid * 2 + 2)

        # 2) Define state & solution
        check_1, check_2 = self.state[nid][fid]
        sol = process_queue(self.kernel_data_2, nid, pid)

        # 3) Assert 247123468
        self.assertEqual(check_1, 0.57) # queue, phase 1
        self.assertEqual(check_1, sol) # queue, phase 1

        # 4) Assert 247123468 prevous cycle
        self.assertEqual(check_2, 0.45) # queue, phase 1, feature 2

    def test_min_queue_squared_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(
            reward[node_id],
            -0.01*((1.05**2 + 0.57**2) - (2.05**2 + 0.45**2)))

    def tearDown(self):
        pass

def process_queue(data, node_id, phase_id):

        def fn(x):
            if (x / 13.89) < 0.1:
                return 1.0
            else:
                return 0.0

        queues = []
        for t in data:

            temp_count = {}
            for veh in t[node_id][phase_id]:

                str_key = veh.edge_id + '_' + str(veh.lane)

                if str_key in temp_count:
                    temp_count[str_key] += fn(veh.speed)
                else:
                    temp_count[str_key] = fn(veh.speed)

            if len(temp_count) == 0:
                queues.append(0.0)
            else:
                queues.append(max(temp_count.values()))


        ret =  round(sum(queues) / 60, 2)

        return ret

if __name__ == '__main__':
    unittest.main()
