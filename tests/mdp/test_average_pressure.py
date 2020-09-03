import unittest

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.utils.aux import flatten
from ilurl.utils.properties import lazy_property

from tests.network.test_grid import *

class TestAveragePressure(TestGridBase):
    """
        * Tests average pressure related state and reward

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
                        features=('average_pressure',),
                        reward='reward_min_average_pressure',
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

            # TODO: create a handler after_update
            # In the case of average pressure -- variable
            # actualization happens on state request
            state = observation_space.feature_map(
                categorize=self.mdp_params.discretize_state_space,
                flatten=True
            )

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

        super(TestAveragePressure, self).setUp()


    # TODO: move this to observation_space test
    # def test_num_phases_247123161(self):
    #     self.assertEqual(len(self.state['247123161']), 2)

    # def test_num_phases_247123464(self):
    #     self.assertEqual(len(self.state['247123464']), 2)

    # def test_num_phases_247123468(self):
    #     self.assertEqual(len(self.state['247123468']), 2)

    # def test_outgoing_247123161_0(self):
    #     # internal outgoing edge for phase 0 
    #     p0 = self.observation_space['247123161']['247123161#0']
    #     test = sorted([outid for outid in p0.outgoing])
    #     sol = INT_OUTGOING_247123161
    #     self.assertEqual(test, sol)

    # def test_outgoing_247123161_1(self):
    #     # internal outgoing edge for phase 0 
    #     p1 = self.observation_space['247123161']['247123161#1']
    #     test = sorted([outid for outid in p1.outgoing])
    #     sol = INT_OUTGOING_247123161
    #     self.assertEqual(test, sol)


    # def test_outgoing_247123464_0(self):
    #     # internal outgoing edge for phase 0 
    #     p0 = self.observation_space['247123464']['247123464#0']
    #     test = sorted([outid for outid in p0.outgoing])
    #     sol = INT_OUTGOING_247123464
    #     self.assertEqual(test, sol)

    # def test_outgoing_247123464_1(self):
    #     # internal outgoing edge for phase 0 
    #     p1 = self.observation_space['247123464']['247123464#1']
    #     test = sorted([outid for outid in p1.outgoing])
    #     sol = INT_OUTGOING_247123464
    #     self.assertEqual(test, sol)


    # def test_outgoing_247123468_0(self):
    #     # internal outgoing edge for phase 0 
    #     p0 = self.observation_space['247123468']['247123468#0']
    #     test = sorted([outid for outid in p0.outgoing])
    #     sol = INT_OUTGOING_247123468
    #     self.assertEqual(test, sol)

    # def test_outgoing_247123468_1(self):
    #     # internal outgoing edge for phase 0 
    #     p1 = self.observation_space['247123468']['247123468#1']
    #     test = sorted([outid for outid in p1.outgoing])
    #     sol = INT_OUTGOING_247123468
    #     self.assertEqual(test, sol)

    def test_avg_pressure_tl1ph0(self):
        """Tests pressure state
            * traffic light 1
            * ID = '247123161'
            * phase 0
        """
        ID = '247123161'

        outgoing = INT_OUTGOING_247123161
        incoming = INCOMING_247123161[0]

        p0 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123161 static assertion
        # avg.pressure, phase 0
        self.assertEqual(self.state[ID][0], 3.73, f'avg.pressure:{ID}\tphase:0') 

        # 247123161 dynamic assertion
        # avg.pressure, phase 0
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    # def process_pressure(self,  incoming, outgoing):

    #     os = State(self.network, self.mdp_params)
    #     os.reset()
    #     timesteps = list(range(1,60)) + [0]

    #     press = 0
    #     n = 0
    #     for t, data in zip(timesteps, self.kernel_data):
    #         os.update(t, data)

    #         dat = get_veh_locations(data)
    #         inc = filter_veh_locations(dat, incoming)
    #         out = filter_veh_locations(dat, outgoing)

    #         avgpress = os.feature_map(
    #             categorize=self.mdp_params.discretize_state_space,
    #             flatten=True
    #         )

    #         press += len(inc) - len(out)

    #         test = avgpress['247123161'][0]
    #         check = round(press / (n + 1), 2)
    #         print(t, test, check)
    #         self.assertEqual(test, check, n + 1)
    #         n += 1

        # import ipdb
        # ipdb.set_trace()
    #     return round(press / 60, 2)
    def test_avg_pressure_tl1ph1(self):
        """Tests pressure state
            * traffic light 1
            * ID = '247123161'
            * phase 1
        """
        ID = '247123161'

        outgoing = INT_OUTGOING_247123161
        incoming = INCOMING_247123161[1]

        p1 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123161 static assertion
        # pressure, phase 1
        self.assertEqual(self.state[ID][1], 1.88)

        # 247123161 dynamic assertion
        # pressure, phase 1
        self.assertEqual(self.state[ID][1], p1)


    def test_min_avg_pressure_tl1(self):
        """Tests pressure reward
            * traffic light 1
            * ID = '247123161'
        """
        ID = '247123161'
        reward = self.reward(self.observation_space)
        self.assertAlmostEqual(reward[ID], -0.01*(3.73 + 1.88))

    # def test_pressure_tl2ph0(self):
    #     """Tests pressure state
    #         * traffic light 2
    #         * ID = '247123464'
    #         * phase 0
    #     """
    #     ID = '247123464'

    #     outgoing = INT_OUTGOING_247123464
    #     incoming = INCOMING_247123464[0]

    #     p0  = process_pressure(self.kernel_data, incoming, outgoing)

    #     # State.
    #     # 247123464 static assertion
    #     self.assertEqual(self.state[ID][0], -3.0) # pressure, phase 0

    #     # 247123464 dynamic assertion
    #     self.assertEqual(self.state[ID][0], p0) # pressure, phase 0

    # def test_pressure_tl2ph1(self):
    #     """Tests pressure state
    #         * traffic light 2
    #         * ID = '247123464'
    #         * phase 1
    #     """
    #     ID = '247123464'

    #     outgoing = INT_OUTGOING_247123464
    #     incoming = INCOMING_247123464[1]

    #     p1  = process_pressure(self.kernel_data, incoming, outgoing)

    #     # State.
    #     # 247123464 static assertion
    #     self.assertEqual(self.state[ID][1], -2.0) # pressure, phase 1

    #     # 247123464 dynamic assertion
    #     self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

    #     # # Reward.
    #     # reward = self.reward(self.observation_space)
    #     # self.assertEqual(reward[ID], 0.01*(3.0 + 2.0))

    # def test_min_pressure_tl2(self):
    #     """Tests pressure reward
    #         * traffic light 2
    #         * ID = '247123464'
    #     """
    #     ID = '247123464'
    #     reward = self.reward(self.observation_space)
    #     self.assertEqual(reward[ID], 0.01*(3.0 + 2.0))

    # def test_pressure_tl3ph0(self):
    #     """Tests pressure state
    #         * traffic light 3
    #         * ID = '247123468'
    #         * phase 0
    #     """
    #     ID = '247123468'

    #     outgoing = INT_OUTGOING_247123468
    #     incoming = INCOMING_247123468[0]

    #     p0 = process_pressure(self.kernel_data, incoming, outgoing)

    #     # State.
    #     # 247123468 static assertion
    #     self.assertEqual(self.state[ID][0], 1.0) # pressure, phase 0

    #     # 247123468 dynamic assertion
    #     self.assertEqual(self.state[ID][0], p0) # pressure, phase 0


    # def test_pressure_tl3ph1(self):
    #     """Tests pressure state
    #         * traffic light 3
    #         * ID = '247123468'
    #         * phase 1
    #     """
    #     ID = '247123468'

    #     outgoing = INT_OUTGOING_247123468
    #     incoming = INCOMING_247123468[1]

    #     p1 = process_pressure(self.kernel_data, incoming, outgoing)

    #     # State.
    #     # 247123468 static assertion
    #     self.assertEqual(self.state[ID][1], 0.0) # pressure, phase 1

    #     # 247123468 dynamic assertion
    #     self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

    # def test_min_pressure_tl3(self):
    #     """Tests pressure reward
    #         * traffic light 3
    #         * ID = '247123468'
    #     """
    #     ID = '247123468'
    #     reward = self.reward(self.observation_space)
    #     self.assertEqual(reward[ID], -0.01*(1.0 + 0.0))

def process_pressure(kernel_data, incoming, outgoing):

    timesteps = list(range(1,60)) + [0]

    press = 0
    for t, data in zip(timesteps, kernel_data):
        dat = get_veh_locations(data)
        inc = filter_veh_locations(dat, incoming)
        out = filter_veh_locations(dat, outgoing)

        press += len(inc) - len(out)

    return round(press / 60, 2)


def get_veh_locations(tl_data):
    """Help flattens hierarchial data

    Params:
    ------
        * tl_data: dict<str, dict<int, list<namedtuple<Vehicle>>>>
            nested dict containing tls x phases x vehicles

    Returns:
    --------
        * veh_locations: list<Tuple>
            list containing triplets: veh_id, edge_id, lane
    """

    # 1) Produces a flat generator with 3 informations: veh_id, edge_id, lane
    gen = flatten([(veh.id, veh.edge_id, veh.lane)
                    for ph_data in tl_data.values()
                    for vehs in ph_data.values()
                    for veh in vehs])

    # 2) generates a list of triplets
    it = iter(gen)
    ret = []
    for x in it:
        ret.append((x, next(it), next(it)))
    return ret

def filter_veh_locations(veh_locations, lane_ids):
    """Help flattens hierarchial data"""
    return [vehloc[0] for vehloc in veh_locations if vehloc[1:] in lane_ids]


if __name__ == '__main__':
    unittest.main()
