import unittest

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.network.test_grid import (TestGridBase,
                                     INCOMING_247123161, OUTGOING_247123161,
                                     INCOMING_247123464, OUTGOING_247123464,
                                     INCOMING_247123468, OUTGOING_247123468,
                                     MAX_VEHS, MAX_VEHS_OUT)

from tests.utils import process_pressure

class TestGridPressure(TestGridBase):
    """
        * Tests pressure related state and reward

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
                        features=('pressure',),
                        reward='reward_min_pressure',
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

        super(TestGridPressure, self).setUp()


    def test_pressure_tl1ph0(self):
        """Tests pressure state
            * traffic light 1
            * ID = '247123161'
            * phase0
        """
        ID = '247123161'

        outgoing = OUTGOING_247123161
        incoming = INCOMING_247123161[0]

        p0 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][0], 5.0, f'pressure:{ID}\tphase:0') # pressure, phase 0

        # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    def test_pressure_tl1ph1(self):
        """Tests pressure state
            * traffic light 1
            * ID = '247123161'
            * phase 1
        """
        ID = '247123161'

        outgoing = OUTGOING_247123161
        incoming = INCOMING_247123161[1]

        p1 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][1], 0.0) # pressure, phase 1

        # # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_pressure_tl1(self):
        """Tests pressure reward
            * traffic light 1
            * ID = '247123161'
        """
        ID = '247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(5.0 + 0.0))

    def test_pressure_tl2ph0(self):
        """Tests pressure state
            * traffic light 2
            * ID = '247123464'
            * phase 0
        """
        ID = '247123464'

        outgoing = OUTGOING_247123464
        incoming = INCOMING_247123464[0]

        p0  = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][0], -3.0) # pressure, phase 0

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    def test_pressure_tl2ph1(self):
        """Tests pressure state
            * traffic light 2
            * ID = '247123464'
            * phase 1
        """
        ID = '247123464'

        outgoing = OUTGOING_247123464
        incoming = INCOMING_247123464[1]

        p1  = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][1], -2.0) # pressure, phase 1

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_pressure_tl2(self):
        """Tests pressure reward
            * traffic light 2
            * ID = '247123464'
        """
        ID = '247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], 0.01*(3.0 + 2.0))

    def test_pressure_tl3ph0(self):
        """Tests pressure state
            * traffic light 3
            * ID = '247123468'
            * phase 0
        """
        ID = '247123468'

        outgoing = OUTGOING_247123468
        incoming = INCOMING_247123468[0]

        p0 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][0], 1.0) # pressure, phase 0

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0


    def test_pressure_tl3ph1(self):
        """Tests pressure state
            * traffic light 3
            * ID = '247123468'
            * phase 1
        """
        ID = '247123468'

        outgoing = OUTGOING_247123468
        incoming = INCOMING_247123468[1]

        p1 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][1], 0.0) # pressure, phase 1

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

    def test_min_pressure_tl3(self):
        """Tests pressure reward
            * traffic light 3
            * ID = '247123468'
        """
        ID = '247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(1.0 + 0.0))


class TestGridPressureNorm(TestGridPressure):
    """
        * Tests pressure related state and reward

        * Normalize state space by number of vehicles

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
                        features=('pressure',),
                        reward='reward_min_pressure',
                        normalize_velocities=True,
                        normalize_vehicles=self.norm_vehs,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    @property
    def norm_vehs(self):
        return True

    def test_pressure_tl1ph0(self):
        """Tests pressure state
            * traffic light 1
            * ID = '247123161'
            * phase0
        """
        ID = '247123161'

        outgoing = OUTGOING_247123161
        incoming = INCOMING_247123161[0]
        fct1 = MAX_VEHS[(ID, 0)] if self.norm_vehs else 1
        fct2 = MAX_VEHS_OUT[(ID, 0)] if self.norm_vehs else 1

        p0 = process_pressure(self.kernel_data, incoming, outgoing,
                              fctin=fct1, fctout=fct2)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][0], 0.1389, f'pressure:{ID}\tphase:0') # pressure, phase 0

        # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    def test_pressure_tl1ph1(self):
        """Tests pressure state
            * traffic light 1
            * ID = '247123161'
            * phase 1
        """
        ID = '247123161'

        outgoing = OUTGOING_247123161
        incoming = INCOMING_247123161[1]
        fct1 = MAX_VEHS[(ID, 1)] if self.norm_vehs else 1
        fct2 = MAX_VEHS_OUT[(ID, 1)] if self.norm_vehs else 1

        p1 = process_pressure(self.kernel_data, incoming, outgoing,
                              fctin=fct1, fctout=fct2)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][1], 0.0) # pressure, phase 1
        # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_pressure_tl1(self):
        """Tests pressure reward
            * traffic light 1
            * ID = '247123161'
        """
        ID = '247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(0.1389 + 0.0))

    def test_pressure_tl2ph0(self):
        """Tests pressure state
            * traffic light 2
            * ID = '247123464'
            * phase 0
        """
        ID = '247123464'

        outgoing = OUTGOING_247123464
        incoming = INCOMING_247123464[0]
        fct1 = MAX_VEHS[(ID, 0)] if self.norm_vehs else 1
        fct2 = MAX_VEHS_OUT[(ID, 0)] if self.norm_vehs else 1

        p0 = process_pressure(self.kernel_data, incoming, outgoing,
                              fctin=fct1, fctout=fct2)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][0], -0.0882) # pressure, phase 0

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    def test_pressure_tl2ph1(self):
        """Tests pressure state
            * traffic light 2
            * ID = '247123464'
            * phase 1
        """
        ID = '247123464'

        outgoing = OUTGOING_247123464
        incoming = INCOMING_247123464[1]

        fct1 = MAX_VEHS[(ID, 1)] if self.norm_vehs else 1
        fct2 = MAX_VEHS_OUT[(ID, 1)] if self.norm_vehs else 1

        p1 = process_pressure(self.kernel_data, incoming, outgoing,
                              fctin=fct1, fctout=fct2)
        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][1], 0.0229) # pressure, phase 1

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_pressure_tl2(self):
        """Tests pressure reward
            * traffic light 2
            * ID = '247123464'
        """
        ID = '247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[ID], -0.01*(-0.0882 + 0.0229))

    def test_pressure_tl3ph0(self):
        """Tests pressure state
            * traffic light 3
            * ID = '247123468'
            * phase 0
        """
        ID = '247123468'

        outgoing = OUTGOING_247123468
        incoming = INCOMING_247123468[0]

        fct1 = MAX_VEHS[(ID, 0)] if self.norm_vehs else 1
        fct2 = MAX_VEHS_OUT[(ID, 0)] if self.norm_vehs else 1

        p0 = process_pressure(self.kernel_data, incoming, outgoing,
                              fctin=fct1, fctout=fct2)
        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][0], 0.0312) # pressure, phase 0

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0


    def test_pressure_tl3ph1(self):
        """Tests pressure state
            * traffic light 3
            * ID = '247123468'
            * phase 1
        """
        ID = '247123468'

        outgoing = OUTGOING_247123468
        incoming = INCOMING_247123468[1]
        fct1 = MAX_VEHS[(ID, 1)] if self.norm_vehs else 1
        fct2 = MAX_VEHS_OUT[(ID, 1)] if self.norm_vehs else 1

        p1 = process_pressure(self.kernel_data, incoming, outgoing,
                              fctin=fct1, fctout=fct2)
        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][1], 0.0) # pressure, phase 1

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

    def test_min_pressure_tl3(self):
        """Tests pressure reward
            * traffic light 3
            * ID = '247123468'
        """
        ID = '247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(0.0312 + 0.0))



if __name__ == '__main__':
    unittest.main()
