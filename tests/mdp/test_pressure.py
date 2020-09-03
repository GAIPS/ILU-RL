import pickle
import unittest

import numpy as np

from ilurl.state import State
from ilurl.envs.elements import build_vehicles
from ilurl.rewards import build_rewards
from ilurl.params import MDPParams
from ilurl.networks.base import Network
from ilurl.utils.aux import flatten

# incoming approaches
INCOMING_247123161 = {0: [('309265401', 0), ('309265401', 1), ('-238059328', 0), ('-238059328', 1)],
                      1: [('383432312', 0), ('-238059324', 0)]}

INCOMING_247123464 = {0: [('309265400', 0), ('309265400', 1), ('-309265401', 0), ('-309265401', 1)],
    1: [('22941893', 0)]}
INCOMING_247123468 = {0: [('309265402', 0), ('309265402', 1), ('-309265400', 0), ('-309265400', 1)], 1: [('23148196', 0)]}

# outgoing approaches
OUTGOING_247123161 = ['-383432312', '-309265401#0', '-309265401#1', '238059324', '238059324', '238059328#0', '238059328#1']
OUTGOING_247123464 = ['3092655395#1', '-309265400#0', '-309265400#1', '309265401#0', '309265401#1']
OUTGOING_247123468 = ['-309265402#0', '-309265402#1', '309265396#1', '309265400#0', '309265400#1']

# internal outgoing approaches; states do not track
INT_OUTGOING_247123161 = [('-309265401', 0), ('-309265401', 1)]
INT_OUTGOING_247123464 = [('-309265400', 0), ('-309265400', 1), ('309265401', 0), ('309265401', 1)]
INT_OUTGOING_247123468 = [('309265400', 0), ('309265400', 1)]

class TestBase(unittest.TestCase):
    """
        * Defines a network and the kernel data.

        * Common base for problem formulation tests.
    """
    def setUp(self):

        network_args = {
            'network_id': 'grid',
            'horizon': 999,
            'demand_type': 'constant',
            'tls_type': 'rl'
        }
        self.network = Network(**network_args)

        with open('tests/data/grid_kernel_data.dat', "rb") as f:
            kernel_data = pickle.load(f)
        self.kernel_data = kernel_data


class TestPressure(TestBase):
    """
        * Tests pressure related state and reward

        Set of tests that target the implemented
        problem formulations, i.e. state and reward
        function definitions.
    """

    def setUp(self):
        """Code here will run before every test"""

        super(TestPressure, self).setUp()

        mdp_params = MDPParams(
                        features=('pressure',),
                        reward='reward_min_pressure',
                        normalize_state_space=True,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)

        self.observation_space = State(self.network, mdp_params)
        self.observation_space.reset()
        self.reward = build_rewards(mdp_params)

        # Fake environment interaction with state object.
        timesteps = list(range(1,60)) + [0]

        for t, data in zip(timesteps, self.kernel_data):
            self.observation_space.update(t, data)

        # Get state.
        self.state = self.observation_space.feature_map(
            categorize=mdp_params.discretize_state_space,
            flatten=True
        )


    def test_kernel_data(self):
        self.assertEqual(len(self.kernel_data), 60)

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

    def test_pressure_tl1(self):
        """ID = '247123161'"""

        ID = '247123161'

        outgoing = INT_OUTGOING_247123161
        incoming = INCOMING_247123161

        p0, p1 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123161 static assertion
        self.assertEqual(self.state[ID][0], 5.0) # pressure, phase 0
        self.assertEqual(self.state[ID][1], 0.0) # pressure, phase 1

        # 247123161 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

        # # 247123464.
        # self.assertEqual(self.state['247123464'][0], 9) # flow, phase 0
        # self.assertEqual(self.state['247123464'][1], 1) # flow, phase 1

        # # 247123468.
        # self.assertEqual(self.state['247123468'][0], 15) # flow, phase 0
        # self.assertEqual(self.state['247123468'][1], 2) # flow, phase 1

        # Reward.
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(5.0 + 0.0))
        # self.assertEqual(reward['247123464'], 0.01*(9.0  + 1.0))
        # self.assertEqual(reward['247123468'], 0.01*(15 + 2))


    def test_pressure_tl2(self):
        """ID = '247123464'"""
        ID = '247123464'

        outgoing = INT_OUTGOING_247123464
        incoming = INCOMING_247123464

        p0, p1 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123464 static assertion
        self.assertEqual(self.state[ID][0], -3.0) # pressure, phase 0
        self.assertEqual(self.state[ID][1], -2.0) # pressure, phase 1

        # 247123464 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

        # Reward.
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], 0.01*(3.0 + 2.0))

    def test_pressure_tl3(self):
        """ID = '247123468'"""
        ID = '247123468'

        outgoing = INT_OUTGOING_247123468
        incoming = INCOMING_247123468

        p0, p1 = process_pressure(self.kernel_data, incoming, outgoing)

        # State.
        # 247123468 static assertion
        self.assertEqual(self.state[ID][0], 1.0) # pressure, phase 0
        self.assertEqual(self.state[ID][1], 0.0) # pressure, phase 1

        # 247123468 dynamic assertion
        self.assertEqual(self.state[ID][0], p0) # pressure, phase 0
        self.assertEqual(self.state[ID][1], p1) # pressure, phase 1

        # Reward.
        reward = self.reward(self.observation_space)
        self.assertEqual(reward[ID], -0.01*(1.0 + 0.0))

def process_pressure(kernel_data, incoming, outgoing):
    timesteps = list(range(1,60)) + [0]

    for t, data in zip(timesteps, kernel_data):
        dat = get_veh_locations(data)
        vehs_inc_0 = filter_veh_locations(dat, incoming[0])
        vehs_inc_1 = filter_veh_locations(dat, incoming[1])

        vehs_out_0 = filter_veh_locations(dat, outgoing)
        vehs_out_1 = filter_veh_locations(dat, outgoing)

        press0 = len(vehs_inc_0) - len(vehs_out_0)
        press1 = len(vehs_inc_1) - len(vehs_out_1)

    return press0, press1


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
