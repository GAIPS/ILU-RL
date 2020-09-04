import unittest

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.utils.aux import flatten
from ilurl.utils.properties import lazy_property

from tests.utils import process_pressure
from tests.network.test_grid import (TestGridBase,
                                     INCOMING_247123161, OUTGOING_247123161,
                                     INCOMING_247123464, OUTGOING_247123464,
                                     INCOMING_247123468, OUTGOING_247123468,
                                     MAX_VEHS, MAX_VEHS_OUT)

class TestGridAveragePressure(TestGridBase):
    """
        * Tests average pressure related state and reward

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    @property
    def mdp_params(self):
        mdp_params = MDPParams(
                        features=('average_pressure',),
                        reward='reward_min_average_pressure',
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

        super(TestGridAveragePressure, self).setUp()



    def test_avg_pressure_tl1ph0(self):
        """Tests average pressure state
            * traffic light 1
            * ID = '247123161'
            * phase 0
        """
        ID = '247123161'

        outgoing = OUTGOING_247123161
        incoming = INCOMING_247123161[0]

        p0 = process_pressure(self.kernel_data, incoming, outgoing, is_average=True)

        # State.
        # 247123161 static assertion
        # avg.pressure, phase 0
        self.assertEqual(self.state[ID][0], 3.73, f'avg.pressure:{ID}\tphase:0')

        # 247123161 dynamic assertion
        # avg.pressure, phase 0
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    def test_avg_pressure_tl1ph1(self):
        """Tests average pressure state
            * traffic light 1
            * ID = '247123161'
            * phase 1
        """
        ID = '247123161'

        outgoing = OUTGOING_247123161
        incoming = INCOMING_247123161[1]

        p1 = process_pressure(self.kernel_data, incoming, outgoing, is_average=True)

        # State.
        # 247123161 static assertion
        # pressure, phase 1
        self.assertEqual(self.state[ID][1], 1.88)

        # 247123161 dynamic assertion
        # pressure, phase 1
        self.assertEqual(self.state[ID][1], p1)


    def test_min_avg_pressure_tl1(self):
        """Tests average pressure reward
            * traffic light 1
            * ID = '247123161'
        """
        ID = '247123161'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[ID], -0.01*(3.73 + 1.88))

    def test_avg_pressure_tl2ph0(self):
        """Tests average pressure state
            * traffic light 2
            * ID = '247123464'
            * phase 0
        """
        ID = '247123464'

        outgoing = OUTGOING_247123464
        incoming = INCOMING_247123464[0]

        p0  = process_pressure(self.kernel_data, incoming, outgoing, is_average=True)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][0], -2.58) # pressure, phase 0

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    def test_avg_pressure_tl2ph1(self):
        """Tests average pressure state
            * traffic light 2
            * ID = '247123464'
            * phase 1
        """
        ID = '247123464'

        outgoing = OUTGOING_247123464
        incoming = INCOMING_247123464[1]

        p1  = process_pressure(self.kernel_data, incoming, outgoing, is_average=True)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][1], -2.95) # pressure, phase 1

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

    def test_min_avg_pressure_tl2(self):
        """Tests avg pressure reward
            * traffic light 2
            * ID = '247123464'
        """
        ID = '247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], 0.01*(2.58 + 2.95))

    def test_avg_pressure_tl3ph0(self):
        """Tests avg pressure state
            * traffic light 3
            * ID = '247123468'
            * phase 0
        """
        ID = '247123468'

        outgoing = OUTGOING_247123468
        incoming = INCOMING_247123468[0]

        p0 = process_pressure(self.kernel_data, incoming, outgoing, is_average=True)

        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][0], 0.73) # pressure, phase 0

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0


    def test_avg_pressure_tl3ph1(self):
        """Tests avg pressure state
            * traffic light 3
            * ID = '247123468'
            * phase 1
        """
        ID = '247123468'

        outgoing = OUTGOING_247123468
        incoming = INCOMING_247123468[1]

        p1 = process_pressure(self.kernel_data, incoming, outgoing, is_average=True)

        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][1], 0.02) # pressure, phase 1

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

    def test_min_pressure_tl3(self):
        """Tests pressure reward
            * traffic light 3
            * ID = '247123468'
        """
        ID = '247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(0.73 + 0.02))


if __name__ == '__main__':
    unittest.main()
