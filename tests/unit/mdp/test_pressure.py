import unittest

from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.mdp.test_mdp_base import (TestGridMDPSetUp,
                                     INCOMING_247123161, OUTGOING_247123161,
                                     INCOMING_247123464, OUTGOING_247123464,
                                     INCOMING_247123468, OUTGOING_247123468,
                                     MAX_VEHS, MAX_VEHS_OUT)
from tests.unit.utils import process_pressure

class TestGridPressure(TestGridMDPSetUp):
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

        p0 = process_pressure(self.kernel_data_1, incoming, outgoing)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][0], 2.0, f'pressure:{ID}\tphase:0') # pressure, phase 0

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

        p1 = process_pressure(self.kernel_data_1, incoming, outgoing)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][1], -1.0) # pressure, phase 1

        # # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_pressure_tl1(self):
        """Tests pressure reward
            * traffic light 1
            * ID = '247123161'
        """
        ID = '247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], round(-0.01*(2.0 -1.0), 4))

    def test_pressure_tl2ph0(self):
        """Tests pressure state
            * traffic light 2
            * ID = '247123464'
            * phase 0
        """
        ID = '247123464'

        outgoing = OUTGOING_247123464
        incoming = INCOMING_247123464[0]

        p0  = process_pressure(self.kernel_data_1, incoming, outgoing)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][0], 0.0) # pressure, phase 0

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

        p1  = process_pressure(self.kernel_data_1, incoming, outgoing)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][1], 3.0) # pressure, phase 1

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_pressure_tl2(self):
        """Tests pressure reward
            * traffic light 2
            * ID = '247123464'
        """
        ID = '247123464'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], round(-0.01*(0.0 + 3.0), 4))

    def test_pressure_tl3ph0(self):
        """Tests pressure state
            * traffic light 3
            * ID = '247123468'
            * phase 0
        """
        ID = '247123468'

        outgoing = OUTGOING_247123468
        incoming = INCOMING_247123468[0]

        p0 = process_pressure(self.kernel_data_1, incoming, outgoing)

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

        p1 = process_pressure(self.kernel_data_1, incoming, outgoing)

        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][1], 1.0) # pressure, phase 1

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

    def test_min_pressure_tl3(self):
        """Tests pressure reward
            * traffic light 3
            * ID = '247123468'
        """
        ID = '247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], round(-0.01*(1.0 + 1.0), 4))


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

        p0 = process_pressure(self.kernel_data_1, incoming, outgoing,
                              fctin=fct1, fctout=fct2)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][0], 0.125, f'pressure:{ID}\tphase:0') # pressure, phase 0

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

        p1 = process_pressure(self.kernel_data_1, incoming, outgoing,
                              fctin=fct1, fctout=fct2)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][1], -0.0972) # pressure, phase 1
        # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_pressure_tl1(self):
        """Tests pressure reward
            * traffic light 1
            * ID = '247123161'
        """
        ID = '247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], round(-0.01*(0.125 - 0.0972), 4))

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

        p0 = process_pressure(self.kernel_data_1, incoming, outgoing,
                              fctin=fct1, fctout=fct2)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][0], 0.0) # pressure, phase 0

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

        p1 = process_pressure(self.kernel_data_1, incoming, outgoing,
                              fctin=fct1, fctout=fct2)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][1], 0.0938) # pressure, phase 1

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_pressure_tl2(self):
        """Tests pressure reward
            * traffic light 2
            * ID = '247123464'
        """
        ID = '247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[ID], round(-0.01*(0.0 +  0.0938), 4))

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

        p0 = process_pressure(self.kernel_data_1, incoming, outgoing,
                              fctin=fct1, fctout=fct2)

        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][0], 0.1597) # pressure, phase 0

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

        p1 = process_pressure(self.kernel_data_1, incoming, outgoing,
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
        self.assertEqual(reward[ID], round(-0.01*(0.1597 + 0.0), 4))


if __name__ == '__main__':
    unittest.main()
