import numpy as np

import unittest

from ilurl.state.state import State
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp

class TestGridMaxFlow(TestGridMDPSetUp):
    """
        * Tests flow wrt Grid network (reward_max_flow)

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
                        features=('flow',),
                        reward='reward_max_flow',
                        normalize_velocities=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridMaxFlow, self).setUp()

    def test_flow_tl1ph0(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_flow(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check, 3.0) # flow, phase 0
        self.assertEqual(check, sol) # flow, phase 0

    def test_flow_tl1ph1(self):
        # 1) Define constraints
        node_id ='247123161'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_flow(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123161
        self.assertEqual(check, 17.0) # flow, phase 1
        self.assertEqual(check, sol) # flow, phase 1

    def test_min_flow_tl1(self):
        node_id ='247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(0.01*(3.0  + 17.0), 4))

    def test_flow_tl2ph0(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_flow(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check, 1.0) # flow, phase 0
        self.assertEqual(check, sol) # flow, phase 0

    def test_flow_tl2ph1(self):
        # 1) Define constraints
        node_id ='247123464'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_flow(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123464
        self.assertEqual(check, 14.0) # flow, phase 1
        self.assertEqual(check, sol) # flow, phase 1

    def test_min_flow_tl2(self):
        node_id ='247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(0.01*(1.0 + 14.0), 4))


    def test_flow_tl3ph0(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 0

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_flow(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check, 1.0) # flow, phase 0
        self.assertEqual(check, sol) # flow, phase 0

    def test_flow_tl3ph1(self):
        # 1) Define constraints
        node_id ='247123468'
        phase_id = 1

        # 2) Define state & solution
        check = self.state[node_id][phase_id]
        sol = process_flow(self.kernel_data_1, node_id, phase_id)

        # 3) Assert 247123468
        self.assertEqual(check, 15.0) # flow, phase 1
        self.assertEqual(check, sol) # flow, phase 1

    def test_min_flow_tl3(self):
        node_id ='247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[node_id], round(0.01*(1.0 + 15.0), 4))

    def tearDown(self):
        pass

def process_flow(kernel_data, node_id, phase_id):
    """Processes batched flow computation"""

    veh_set = set()
    timesteps = list(range(1,60)) + [0]

    for t, data in zip(timesteps, kernel_data):
        if t == 0:
            veh_set_1 = {veh.id for veh in data[node_id][phase_id]}
        else:
            veh_set = veh_set.union({veh.id for veh in data[node_id][phase_id]})

    ret = len(veh_set - veh_set_1)
    ret = round(ret, 2)
    return ret

if __name__ == '__main__':
    unittest.main()