import numpy as np

import unittest
from collections import defaultdict

from ilurl.state.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.utils.aux import flatten
from ilurl.utils.properties import lazy_property

from tests.unit.network.test_grid import MAX_VEHS_PER_LANE
from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp

class TestGridWaitingTime(TestGridMDPSetUp):
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

    def setUp(self):
        """Code here will run before every test"""

        super(TestGridWaitingTime, self).setUp()


    def test_wait_tl1ph0(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check, 1.3) # wait_t, phase 0
        self.assertEqual(check, sol) # wait_t, phase 0

    def test_wait_t_tl1ph1(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check, 1.32) # wait_t, phase 1
        self.assertEqual(check, sol) # wait_t, phase 1

    def test_min_wait_t_tl1(self):
        node_id ='247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(1.3 + 1.32), 4))

    def test_wait_t_tl2ph0(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check, 0.00) # wait_t, phase 0
        self.assertEqual(check, sol) # wait_t, phase 0

    def test_wait_t_tl2ph1(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check, 0.6) # wait_t, phase 1
        self.assertEqual(check, sol) # wait_t, phase 1

    def test_min_wait_t_tl2(self):
        node_id ='247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(0.0 + 0.6), 4))


    def test_wait_t_tl3ph0(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check,  0.82) # wait_t, phase 0
        self.assertEqual(check, sol) # wait_t, phase 0

    def test_wait_t_tl3ph1(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check, 0.83) # wait_t, phase 1
        self.assertEqual(check, sol) # wait_t, phase 1

    def test_min_wait_t_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(0.82 + 0.83), 4))

    def tearDown(self):
        pass


class TestGridWaitingTimeNorm(TestGridMDPSetUp):
    """
        * Tests waiting time wrt Grid network

        * Normalize vehicles count

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    @property
    def norm_vehs(self):
        return True

    @lazy_property
    def mdp_params(self):
        mdp_params = MDPParams(
                        features=('waiting_time',),
                        reward='reward_min_waiting_time',
                        normalize_velocities=True,
                        normalize_vehicles=self.norm_vehs,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""

        super(TestGridWaitingTimeNorm, self).setUp()

    def test_wait_t_tl1ph0(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id, norm_vehs=self.norm_vehs)

        # 3) Assert 247123161
        self.assertEqual(check, 0.16) # wait_t, phase 0
        self.assertEqual(check, sol) # wait_t, phase 0

    def test_wait_t_tl1ph1(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id,
                            norm_vehs=self.norm_vehs)

        # 3) Assert 247123161
        self.assertEqual(check, 0.15) # wait_t, phase 1
        self.assertEqual(check, sol) # wait_t, phase 1

    def test_min_wait_t_tl1(self):
        node_id ='247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(0.16 + 0.15), 4))

    def test_wait_t_tl2ph0(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id,
                        norm_vehs=self.norm_vehs)

        # 3) Assert 247123464
        self.assertEqual(check, 0.00) # wait_t, phase 0
        self.assertEqual(check, sol) # wait_t, phase 0

    def test_wait_t_tl2ph1(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id,
                            norm_vehs=self.norm_vehs)

        # 3) Assert 247123464
        self.assertEqual(check, 0.07) # wait_t, phase 1
        self.assertEqual(check, sol) # wait_t, phase 1

    def test_min_wait_t_tl2(self):
        node_id ='247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(0.0 + 0.07), 4))


    def test_wait_t_tl3ph0(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id,
                            norm_vehs=self.norm_vehs)

        # 3) Assert 247123468
        self.assertEqual(check,  0.09) # wait_t, phase 0
        self.assertEqual(check, sol) # wait_t, phase 0

    def test_wait_t_tl3ph1(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_waiting_time(self.kernel_data_1, node_id, phase_id,
                            norm_vehs=self.norm_vehs)

        # 3) Assert 247123468
        self.assertEqual(check, 0.1) # wait_t, phase 1
        self.assertEqual(check, sol) # wait_t, phase 1

    def test_min_wait_t_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(-0.01*(0.09 + 0.1), 4))

    def tearDown(self):
        pass


def process_waiting_time(kernel_data, node_id, phase_id, norm_vehs=False):
    """Processes batched waiting time computation"""
    cycle_time = 60

    def fn(x):
        if (x / 13.89) < 0.1:
            return 1.0
        else:
            return 0.0

    wait_times = []
    for t in kernel_data:

        qt = defaultdict(lambda : 0)
        for veh in t[node_id][phase_id]:

            key = (veh.edge_id, veh.lane)
            qt[key] += fn(veh.speed)

        if len(qt) == 0:
            wait_times.append(0.0)
        else:
            if norm_vehs:
                wait_times.append(
                    sum([v / MAX_VEHS_PER_LANE[k]  for k, v in qt.items()]))
            else:
                wait_times.append(sum(qt.values()))

    ret =  round(sum(wait_times) / cycle_time, 2)

    return ret

if __name__ == '__main__':
    unittest.main()
