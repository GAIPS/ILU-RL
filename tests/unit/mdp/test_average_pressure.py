import unittest

from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.utils import process_pressure
from tests.unit.mdp.test_mdp_base import (TestGridMDPSetUp,
                                     INCOMING_247123161, OUTGOING_247123161,
                                     INCOMING_247123464, OUTGOING_247123464,
                                     INCOMING_247123468, OUTGOING_247123468,
                                     MAX_VEHS, MAX_VEHS_OUT)

class TestGridAveragePressure(TestGridMDPSetUp):
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

    def test_min_avg_pressure_tl3(self):
        """Tests pressure reward
            * traffic light 3
            * ID = '247123468'
        """
        ID = '247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(0.73 + 0.02))

    def tearDown(self):
        pass

class TestGridAveragePressureNorm(TestGridAveragePressure):
    """
        * Tests average pressure related state and reward

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
                        features=('average_pressure',),
                        reward='reward_min_average_pressure',
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

    def test_avg_pressure_tl1ph0(self):
        """Tests avg.pressure state
            * traffic light 1
            * ID = '247123161'
            * phase 0
        """
        ID = '247123161'

        outgoing = OUTGOING_247123161
        incoming = INCOMING_247123161[0]
        fct1 = MAX_VEHS[(ID, 0)] if self.norm_vehs else 1
        fct2 = MAX_VEHS_OUT[(ID, 0)] if self.norm_vehs else 1

        p0 = process_pressure(self.kernel_data, incoming, outgoing,
                              fctin=fct1, fctout=fct2, is_average=True)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][0], 0.1) # pressure, phase 0

        # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    def test_avg_pressure_tl1ph1(self):
        """Tests avg.pressure state
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
                              fctin=fct1, fctout=fct2, is_average=True)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][1], 0.12) # pressure, phase 1
        # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_avg_pressure_tl1(self):
        """Tests pressure reward
            * traffic light 1
            * ID = '247123161'
        """
        ID = '247123161'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(0.1 + 0.12))


    def test_avg_pressure_tl2ph0(self):
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
                              fctin=fct1, fctout=fct2, is_average=True)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][0], -0.07) # pressure, phase 0

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    def test_avg_pressure_tl2ph1(self):
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
                              fctin=fct1, fctout=fct2, is_average=True)
        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][1], -0.06) # pressure, phase 1

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1


    def test_min_avg_pressure_tl2(self):
        """Tests pressure reward
            * traffic light 2
            * ID = '247123464'
        """
        ID = '247123464'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[ID], -0.01*(-0.07 - 0.06))

    def test_avg_pressure_tl3ph0(self):
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
                              fctin=fct1, fctout=fct2, is_average=True)
        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][0], 0.01) # pressure, phase 0

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0


    def test_avg_pressure_tl3ph1(self):
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
                              fctin=fct1, fctout=fct2, is_average=True)
        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][1], 0.03) # pressure, phase 1

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

    def test_min_avg_pressure_tl3(self):
        """Tests pressure reward
            * traffic light 3
            * ID = '247123468'
        """
        ID = '247123468'
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(0.03 + 0.01))

if __name__ == '__main__':
    unittest.main()
