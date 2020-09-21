import unittest


from ilurl.rewards import build_rewards
from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.network.test_grid import (TestGridSetUp,
                                     MAX_VEHS, MAX_VEHS_OUT,
                                     INCOMING_247123161, OUTGOING_247123161,
                                     INCOMING_247123464, OUTGOING_247123464,
                                     INCOMING_247123468, OUTGOING_247123468,
                                     MAX_VEHS, MAX_VEHS_OUT)

class TestGridMDPSetUp(TestGridSetUp):
    """
        * Sets up common problem formulations

        * Tests here will run also on children

        * Avoid this behaviour by placing observation
          space tests in another class.
    """

    @lazy_property
    def observation_space(self):
        observation_space = State(self.network, self.mdp_params)
        observation_space.reset()
        # Fake environment interaction with state object.
        timesteps = list(range(1,60)) + [0]

        for t, data in zip(timesteps, self.kernel_data):
            observation_space.update(t, *data)

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

        super(TestGridMDPSetUp, self).setUp()


class TestGridTLS1ObservationSpace(TestGridMDPSetUp):
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
                        features=('waiting_time',),
                        reward='reward_min_waiting_time',
                        normalize_velocities=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridTLS1ObservationSpace, self).setUp()

        self.ID = '247123161'
        self.STATE = self.state[self.ID]
        self.PHASE_0 = self.observation_space[self.ID][f'{self.ID}#0']
        self.PHASE_1 = self.observation_space[self.ID][f'{self.ID}#1']
        self.OUTGOING = OUTGOING_247123161

    def test_num_phases(self):
        self.assertEqual(len(self.STATE), 4)

    def test_outgoing_0(self):
        # internal outgoing edge for phase 0 
        test = sorted([outid for outid in self.PHASE_0.outgoing])
        self.assertEqual(test, self.OUTGOING)

    def test_outgoing_1(self):
        # internal outgoing edge for phase 0 
        test = sorted([outid for outid in self.PHASE_1.outgoing])
        self.assertEqual(test, self.OUTGOING)

    def test_tl1_0_green(self):
        self.assertEqual(self.PHASE_0.green_states, [['R', 'G'], ['R', 'Y']])

    def test_tl1_0_max_vehs(self):
        check = self.PHASE_0.max_vehs
        sol = MAX_VEHS[(self.ID, 0)]
        self.assertEqual(check, 36)
        self.assertEqual(check, sol)

    def test_tl1_1_max_vehs(self):
        check = self.PHASE_1.max_vehs
        sol = MAX_VEHS[(self.ID, 1)]
        self.assertEqual(check, 16)
        self.assertEqual(check, sol)

    def test_tl1_0_max_vehs_out(self):
        check = self.PHASE_0.max_vehs_out
        sol = MAX_VEHS_OUT[(self.ID, 0)]
        self.assertEqual(check, 16)
        self.assertEqual(check, sol)

    def test_tl1_1_green(self):
        self.assertEqual(self.PHASE_1.green_states, [['G', 'R'], ['Y', 'R']])

    def test_tl1_1_max_vehs_out(self):
        check = self.PHASE_1.max_vehs_out
        sol = MAX_VEHS_OUT[(self.ID, 1)]
        self.assertEqual(check, 16)
        self.assertEqual(check, sol)

class TestGridTLS2ObservationSpace(TestGridMDPSetUp):
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
                        features=('waiting_time',),
                        reward='reward_min_waiting_time',
                        normalize_velocities=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridTLS2ObservationSpace, self).setUp()

        self.ID = '247123464'
        self.STATE = self.state[self.ID]
        self.PHASE_0 = self.observation_space[self.ID][f'{self.ID}#0']
        self.PHASE_1 = self.observation_space[self.ID][f'{self.ID}#1']
        self.OUTGOING = OUTGOING_247123464

    def test_num_phases(self):
        self.assertEqual(len(self.STATE), 4)

    def test_outgoing_0(self):
        # internal outgoing edge for phase 0 
        test = sorted([outid for outid in self.PHASE_0.outgoing])
        self.assertEqual(test, self.OUTGOING)

    def test_outgoing_1(self):
        # internal outgoing edge for phase 0 
        test = sorted([outid for outid in self.PHASE_1.outgoing])
        self.assertEqual(test, self.OUTGOING)

    def test_tl2_0_green(self):
        self.assertEqual(self.PHASE_0.green_states, [['G', 'R'], ['Y', 'R']])

    def test_tl2_1_green(self):
        self.assertEqual(self.PHASE_1.green_states, [['R', 'G'], ['R', 'Y']])

    def test_tl2_0_max_vehs(self):
        check = self.PHASE_0.max_vehs
        sol = MAX_VEHS[(self.ID, 0)]
        self.assertEqual(check, 32)
        self.assertEqual(check, sol)

    def test_tl2_1_max_vehs(self):
        check = self.PHASE_1.max_vehs
        sol = MAX_VEHS[(self.ID, 1)]
        self.assertEqual(check, 9)
        self.assertEqual(check, sol)

    def test_tl2_0_max_vehs_out(self):
        check = self.PHASE_0.max_vehs_out
        sol = MAX_VEHS_OUT[(self.ID, 0)]
        self.assertEqual(check, 34)
        self.assertEqual(check, sol)

    def test_tl2_1_max_vehs_out(self):
        check = self.PHASE_1.max_vehs_out
        sol = MAX_VEHS_OUT[(self.ID, 1)]
        self.assertEqual(check, 34)
        self.assertEqual(check, sol)

class TestGridTLS3ObservationSpace(TestGridMDPSetUp):
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
                        features=('waiting_time',),
                        reward='reward_min_waiting_time',
                        normalize_velocities=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridTLS3ObservationSpace, self).setUp()

        self.ID = '247123468'
        self.STATE = self.state[self.ID]
        self.PHASE_0 = self.observation_space[self.ID][f'{self.ID}#0']
        self.PHASE_1 = self.observation_space[self.ID][f'{self.ID}#1']
        self.OUTGOING = OUTGOING_247123464

    def test_num_phases(self):
        self.assertEqual(len(self.STATE), 4)

    def test_outgoing_0(self):
        # internal outgoing edge for phase 0 
        test = sorted([outid for outid in self.PHASE_0.outgoing])
        sol = OUTGOING_247123468
        self.assertEqual(test, sol)

    def test_outgoing_1(self):
        # internal outgoing edge for phase 0 
        test = sorted([outid for outid in self.PHASE_1.outgoing])
        sol = OUTGOING_247123468
        self.assertEqual(test, sol)


    def test_tl3_1_green(self):
        self.assertEqual(self.PHASE_1.green_states, [['G', 'R'], ['Y', 'R']])

    def test_tl3_0_green(self):
        self.assertEqual(self.PHASE_0.green_states, [['R', 'G'], ['R', 'Y']])

    def test_tl3_0_max_vehs(self):
        check = self.PHASE_0.max_vehs
        sol = MAX_VEHS[(self.ID, 0)]
        self.assertEqual(check, 32)
        self.assertEqual(check, sol)

    def test_tl3_1_max_vehs(self):
        check = self.PHASE_1.max_vehs
        sol = MAX_VEHS[(self.ID, 1)]
        self.assertEqual(check, 9)
        self.assertEqual(check, sol)


    def test_tl3_0_max_vehs_out(self):
        check = self.PHASE_0.max_vehs_out
        sol = MAX_VEHS_OUT[(self.ID, 0)]
        self.assertEqual(check, 16)
        self.assertEqual(check, sol)

    def test_tl3_1_max_vehs_out(self):
        check = self.PHASE_1.max_vehs_out
        sol = MAX_VEHS_OUT[(self.ID, 1)]
        self.assertEqual(check, 16)
        self.assertEqual(check, sol)

if __name__ == '__main__':
    unittest.main()
