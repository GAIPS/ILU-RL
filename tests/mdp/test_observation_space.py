import unittest

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.network.test_grid import *

class TestObservationSpace(TestGridBase):
    """
        * Tests basic observation space

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
                        features=('pressure',),
                        reward='reward_min_pressure',
                        normalize_state_space=True,
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
    def state(self):
        # Get state.
        state = self.observation_space.feature_map(
            categorize=self.mdp_params.discretize_state_space,
            flatten=True
        )
        return state

    def setUp(self):
        """Code here will run before every test"""

        super(TestObservationSpace, self).setUp()


    def test_num_phases_247123161(self):
        self.assertEqual(len(self.state['247123161']), 2)

    def test_num_phases_247123464(self):
        self.assertEqual(len(self.state['247123464']), 2)

    def test_num_phases_247123468(self):
        self.assertEqual(len(self.state['247123468']), 2)

    def test_outgoing_247123161_0(self):
        # internal outgoing edge for phase 0 
        p0 = self.observation_space['247123161']['247123161#0']
        test = sorted([outid for outid in p0.outgoing])
        sol = INT_OUTGOING_247123161
        self.assertEqual(test, sol)

    def test_outgoing_247123161_1(self):
        # internal outgoing edge for phase 0 
        p1 = self.observation_space['247123161']['247123161#1']
        test = sorted([outid for outid in p1.outgoing])
        sol = INT_OUTGOING_247123161
        self.assertEqual(test, sol)


    def test_outgoing_247123464_0(self):
        # internal outgoing edge for phase 0 
        p0 = self.observation_space['247123464']['247123464#0']
        test = sorted([outid for outid in p0.outgoing])
        sol = INT_OUTGOING_247123464
        self.assertEqual(test, sol)

    def test_outgoing_247123464_1(self):
        # internal outgoing edge for phase 0 
        p1 = self.observation_space['247123464']['247123464#1']
        test = sorted([outid for outid in p1.outgoing])
        sol = INT_OUTGOING_247123464
        self.assertEqual(test, sol)


    def test_outgoing_247123468_0(self):
        # internal outgoing edge for phase 0 
        p0 = self.observation_space['247123468']['247123468#0']
        test = sorted([outid for outid in p0.outgoing])
        sol = INT_OUTGOING_247123468
        self.assertEqual(test, sol)

    def test_outgoing_247123468_1(self):
        # internal outgoing edge for phase 0 
        p1 = self.observation_space['247123468']['247123468#1']
        test = sorted([outid for outid in p1.outgoing])
        sol = INT_OUTGOING_247123468
        self.assertEqual(test, sol)

if __name__ == '__main__':
    unittest.main()
