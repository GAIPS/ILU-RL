import numpy as np

import unittest

from ilurl.state import State
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp

class TestGridTimePeriodSyncVersion(TestGridMDPSetUp):
    """
        * Tests time period variable wrt Grid network.
            (time period is synchronous in comparison
            to the cycle length)

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
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=3600,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridTimePeriodSyncVersion, self).setUp()

    def test_init_state_tl1_ph0(self):
        node_id ='247123161'
        phase_id = 0

        check = self.state[node_id][phase_id+1]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        self.assertEqual(check, 2.91)
        self.assertEqual(check, sol)

    def test_init_state_tl1_ph1(self):
        node_id ='247123161'
        phase_id = 1

        check = self.state[node_id][phase_id+1]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        self.assertEqual(check, 1.23)
        self.assertEqual(check, sol)

    def test_init_state_tl2_ph0(self):
        node_id ='247123464'
        phase_id = 0

        check = self.state[node_id][phase_id+1]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        self.assertEqual(check, 0.02)
        self.assertEqual(check, sol)

    def test_init_state_tl2_ph1(self):
        node_id ='247123464'
        phase_id = 1

        check = self.state[node_id][phase_id+1]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        self.assertEqual(check, 0.09)
        self.assertEqual(check, sol)

    def test_init_state_tl3_ph0(self):
        node_id ='247123468'
        phase_id = 0

        check = self.state[node_id][phase_id+1]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        self.assertEqual(check, 0.69)
        self.assertEqual(check, sol)

    def test_init_state_tl3_ph1(self):
        node_id ='247123468'
        phase_id = 1

        check = self.state[node_id][phase_id+1]
        sol = process_delay(self.kernel_data, node_id, phase_id)

        self.assertEqual(check, 0.29)
        self.assertEqual(check, sol)

    def test_time_period_sync(self):

        self.observation_space.reset()

        hours = list(range(24)) + [0,1]
        for hour in hours:
            for minute in range(60):

                # Fake environment interaction with state object.
                # (60 seconds = 1 minute).
                timesteps = list(range(1,60)) + [0]
                for t, data in zip(timesteps, self.kernel_data):
                    self.observation_space.update(t, data)

                # Get state.
                state = self.observation_space.feature_map(
                    categorize=self.mdp_params.discretize_state_space,
                    flatten=True
                )

                # Assert time period variable.
                self.assertEqual(state['247123161'][0], hour)
                self.assertEqual(state['247123464'][0], hour)
                self.assertEqual(state['247123468'][0], hour)


class TestGridTimePeriodAsyncVersion(TestGridMDPSetUp):
    """
        * Tests time period variable wrt Grid network.
            (time period is not synchronous in comparison
            to the cycle length)

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
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=3599,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridTimePeriodAsyncVersion, self).setUp()

    def test_time_period_async(self):

        self.observation_space.reset()

        hours = list(range(24)) + [0,1]
        for hour in hours:
            for minute in range(1,61):

                # Fake environment interaction with state object.
                # (60 seconds = 1 minute).
                timesteps = list(range(1,60)) + [0]
                for t, data in zip(timesteps, self.kernel_data):
                    self.observation_space.update(t, data)

                # Get state.
                state = self.observation_space.feature_map(
                    categorize=self.mdp_params.discretize_state_space,
                    flatten=True
                )

                seconds_elapsed = hour * (3600) + 60 * (minute)
                t_period = (seconds_elapsed % 86400) // 3599

                # Assert time period variable.
                self.assertEqual(state['247123161'][0], t_period)
                self.assertEqual(state['247123464'][0], t_period)
                self.assertEqual(state['247123468'][0], t_period)


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