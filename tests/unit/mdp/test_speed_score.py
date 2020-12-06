import numpy as np

import unittest

from ilurl.state.state import State
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp

class TestGridSpeedScore(TestGridMDPSetUp):
    """
        * Tests speed score wrt Grid network (reward_max_speed_score)

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
                        features=('speed_score', 'count'),
                        reward='reward_max_speed_score',
                        normalize_velocities=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridSpeedScore, self).setUp()

    def test_speed_count_tl1ph0(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 0

        # 2) Define state & solution
        check_s = self.state[node_id][0]
        check_c = self.state[node_id][1]
        sol = process_speed_score(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check_s, 0.18) # speed_score, phase 0
        self.assertEqual(check_s, sol[0]) # speed_score, phase 0
        self.assertEqual(check_c, 1.93) # count, phase 0
        self.assertEqual(check_c, sol[1]) # count, phase 0

    def test_speed_count_tl1ph1(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 1

        # 2) Define state & solution
        check_s = self.state[node_id][2]
        check_c = self.state[node_id][3]
        sol = process_speed_score(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check_s, 0.42) # speed_score, phase 1
        # self.assertAlmostEqual(check_s, sol[0]) # speed_score, phase 1
        self.assertEqual(check_c, 3.12) # count, phase 1
        self.assertEqual(check_c, sol[1]) # count, phase 1

    def test_min_speed_count_tl1(self):
        node_id ='247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(0.01*(0.18*1.93 + 0.42*3.12), 4))

    def test_speed_count_tl2ph0(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 0

        # 2) Define state & solution
        check_s = self.state[node_id][0]
        check_c = self.state[node_id][1]
        sol = process_speed_score(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check_s, 0.39) # speed_score, phase 0
        self.assertEqual(check_s, sol[0]) # speed_score, phase 0
        self.assertEqual(check_c, 0.05) # count, phase 0
        self.assertEqual(check_c, sol[1]) # count, phase 0

    def test_speed_count_tl2ph1(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 1

        # 2) Define state & solution
        check_s = self.state[node_id][2]
        check_c = self.state[node_id][3]
        sol = process_speed_score(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check_s, 0.57) # speed_score, phase 1
        self.assertEqual(check_s, sol[0]) # speed_score, phase 1
        self.assertEqual(check_c, 1.93) # count, phase 1
        self.assertEqual(check_c, sol[1]) # count, phase 1

    def test_min_speed_count_tl2(self):
        node_id ='247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(0.01*(0.39*0.05 + 0.57*1.93), 4))


    def test_speed_count_tl3ph0(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 0

        # 2) Define state & solution
        check_s = self.state[node_id][0]
        check_c = self.state[node_id][1]
        sol = process_speed_score(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check_s, 0.16) # speed_score, phase 0
        #self.assertEqual(check_s, sol[0]) # speed_score, phase 0
        self.assertEqual(check_c, 1.13) # count, phase 0
        self.assertEqual(check_c, sol[1]) # count, phase 0

    def test_speed_count_tl3ph1(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 1

        # 2) Define state & solution
        check_s = self.state[node_id][2]
        check_c = self.state[node_id][3]
        sol = process_speed_score(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check_s, 0.43) # speed_score, phase 1
        self.assertEqual(check_s, sol[0]) # speed_score, phase 1
        self.assertEqual(check_c,2.18) # count, phase 1
        self.assertEqual(check_c, sol[1]) # count, phase 1

    def test_min_speed_count_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(0.01*(0.16*1.13 + 0.43*2.18), 4))

    def tearDown(self):
        pass


def process_speed_score(kernel_data, node_id, phase_id):
    """Processes batched speed score computation"""
    cycle_time = 60

    values_count = []
    for t in kernel_data:
        values_count.extend(t[node_id][phase_id])

    vehs_speeds = []
    for veh in values_count:
        vehs_speeds.append(veh.speed)

    vehs_speeds = np.array(vehs_speeds)

    count = len(vehs_speeds) / cycle_time
    speed = np.sum(vehs_speeds) / (13.89 * len(vehs_speeds))

    return (round(speed, 2), round(count, 2))


if __name__ == '__main__':
    unittest.main()
