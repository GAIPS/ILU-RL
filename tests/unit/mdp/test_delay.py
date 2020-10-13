from collections import defaultdict
import unittest

import numpy as np

from ilurl.state import State
from ilurl.params import MDPParams
from ilurl.utils.properties import lazy_property

from tests.unit.mdp.test_mdp_base import TestGridMDPSetUp

class TestGridDelaySetUp(TestGridMDPSetUp):
    """
        * Tests delay wrt Grid network (reward_min_delay)

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
                        normalize_vehicles=False,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridDelaySetUp, self).setUp()

class TestGridTLS1DelaySetUp(TestGridDelaySetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS1DelaySetUp, self).setUp()
        self.ID = '247123161'

class TestGridTLS1Delay(TestGridTLS1DelaySetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS1Delay, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_0)

        # 3) Assert 247123161
        self.assertEqual(self.STATE_0, [0.02, 3.2]) # phase 0
        self.assertEqual(self.STATE_0, sol) # phase 0

    def test_1(self):
        # 2) Define state & solution
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_1)

        # 3) Assert 247123161
        self.assertEqual(self.STATE_1, [0.06, 1.22]) # phase 0
        self.assertEqual(self.STATE_1, sol) # phase 0

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (0.02 + 3.2 + 0.06 + 1.22))

class TestGridTLS2DelaySetUp(TestGridDelaySetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS2DelaySetUp, self).setUp()
        self.ID = '247123464'

class TestGridTLS2Delay(TestGridTLS2DelaySetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS2Delay, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_0)

        # 3) Assert 247123464
        self.assertEqual(self.STATE_0, [0.75, 0.01]) # wait_t, phase 0
        self.assertEqual(self.STATE_0, sol) # wait_t, phase 0

    def test_1(self):
        # 2) Define state & solution
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_1)

        # 3) Assert 247123464
        self.assertEqual(self.STATE_1, [0.81, 0.01]) # wait_t, phase 1
        self.assertEqual(self.STATE_1, sol) # wait_t, phase 1

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (0.75 + 0.01 + 0.81 + 0.01))

class TestGridTLS3DelaySetUp(TestGridDelaySetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS3DelaySetUp, self).setUp()
        self.ID = '247123468'

class TestGridTLS3Delay(TestGridTLS3DelaySetUp):
    """
        * Tests waiting time wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS3Delay, self).setUp()
        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0][0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1][0]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_0)

        self.assertEqual(self.STATE_0, [0.09, 1.08]) # wait_t, phase 0
        self.assertEqual(self.STATE_0, sol) # wait_t, phase 0

    def test_1(self):
        sol = process_delay(self.kernel_data, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1, [0.13, 0.74]) # wait_t, phase 1
        self.assertEqual(self.STATE_1, sol) # wait_t, phase 1

    def test_reward(self):
        self.assertAlmostEqual(self.REWARD, -0.01 * (0.09 + 1.08 + 0.13 + 0.74))

    def tearDown(self):
        pass


class TestGridDelayReduxSetUp(TestGridMDPSetUp):
    """
        * Tests delay wrt Grid network (reward_max_delay_reduction)
        * (part 1)

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
                        features=('delay', 'lag[delay]'),
                        reward='reward_max_delay_reduction',
                        normalize_velocities=True,
                        normalize_vehicles=False,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    @lazy_property
    def observation_space(self):
        observation_space = State(self.network, self.mdp_params)
        observation_space.reset()
        for data in self.kernel_data_1:
            observation_space.update(*data)

        return observation_space

    def setUp(self):
        """Code here will run before every test"""

        super(TestGridDelayReduxSetUp, self).setUp()

class TestGridTLS1DelayReduxSetUp(TestGridDelayReduxSetUp):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS1DelayReduxSetUp, self).setUp()
        self.ID = '247123161'

        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]



class TestGridTLS1DelayRedux(TestGridTLS1DelayReduxSetUp):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridTLS1DelayRedux, self).setUp()

    def test_0(self):
        sol = process_delay(self.kernel_data_1, self.ID, self.PHASE_0)

        self.assertEqual(self.STATE_0[0], [0.3, 0.01])
        self.assertEqual(self.STATE_0[0], sol)

    def test_1(self):
        sol = process_delay(self.kernel_data_1, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1[0], [0.52, 1.36])
        self.assertEqual(self.STATE_1[0], sol)

    def test_rewards(self):
        reward = round(-0.01 * ((0.3 - 0) + (0.01 - 0) + (0.52 - 0.0) + (1.36 - 0)), 4)
        self.assertAlmostEqual(self.REWARD, reward)

class TestGridTLS2DelayReduxSetUp(TestGridDelayReduxSetUp):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS2DelayReduxSetUp, self).setUp()
        self.ID = '247123464'

        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]



class TestGridTLS2DelayRedux(TestGridTLS2DelayReduxSetUp):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridTLS2DelayRedux, self).setUp()

    def test_0(self):
        sol = process_delay(self.kernel_data_1, self.ID, self.PHASE_0)

        self.assertEqual(self.STATE_0[0], [0.38, 0.1])
        self.assertEqual(self.STATE_0[0], sol)

    def test_1(self):
        sol = process_delay(self.kernel_data_1, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1[0], [0.17, 0.03])
        self.assertEqual(self.STATE_1[0], sol)

    def test_rewards(self):
        reward = round(-0.01 * ((0.38 - 0) + (0.1 - 0) + (0.17 - 0.0) + (0.03 - 0)), 4)
        self.assertAlmostEqual(self.REWARD, reward)


class TestGridTLS3DelayReduxSetUp(TestGridDelayReduxSetUp):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS3DelayReduxSetUp, self).setUp()
        self.ID = '247123468'

        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]



class TestGridTLS3DelayRedux(TestGridTLS3DelayReduxSetUp):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """

    def setUp(self):
        """Code here will run before every test"""
        super(TestGridTLS3DelayRedux, self).setUp()

    def test_0(self):
        sol = process_delay(self.kernel_data_1, self.ID, self.PHASE_0)

        self.assertEqual(self.STATE_0[0], [0.06, 0.09])
        self.assertEqual(self.STATE_0[0], sol)

    def test_1(self):
        sol = process_delay(self.kernel_data_1, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1[0], [0.22, 1.55])
        self.assertEqual(self.STATE_1[0], sol)

    def test_rewards(self):
        reward = round(-0.01 * ((0.06 - 0) + (0.09 - 0) + (0.22 - 0.0) + (1.55 - 0)), 4)
        self.assertAlmostEqual(self.REWARD, reward)


class TestGridDelayReduxSetUp2(TestGridDelayReduxSetUp):
    """
        * Tests delay wrt Grid network (reward_max_delay_reduction)
        * (part 1)

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
                        features=('delay', 'lag[delay]'),
                        reward='reward_max_delay_reduction',
                        normalize_velocities=True,
                        normalize_vehicles=False,
                        discretize_state_space=False,
                        reward_rescale=0.01,
                        time_period=None,
                        velocity_threshold=0.1)
        return mdp_params

    @lazy_property
    def observation_space(self):
        observation_space = super(TestGridDelayReduxSetUp2, self).observation_space
        for data in self.kernel_data_2:
            observation_space.update(*data)
        return observation_space

    def setUp(self):
        """Code here will run before every test"""

        super(TestGridDelayReduxSetUp2, self).setUp()

class TestGridTLS1DelayReduxSetUp2(TestGridDelayReduxSetUp2):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS1DelayReduxSetUp2, self).setUp()
        self.ID = '247123161'

        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data_2, self.ID, self.PHASE_0)

        self.assertEqual(self.STATE_0[0], [0.05, 2.27])
        self.assertEqual(self.STATE_0[0], sol)

    def test_1(self):
        sol = process_delay(self.kernel_data_2, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1[0], [0.05, 3.15])
        self.assertEqual(self.STATE_1[0], sol)

    def test_rewards(self):
        reward = round(-0.01 * ((0.05 - 0.3) + (2.27 - 0.01) + (0.05 - 0.52) + (3.15 - 1.36)), 4)
        self.assertAlmostEqual(self.REWARD, reward)

class TestGridTLS2DelayRedux2(TestGridDelayReduxSetUp2):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS2DelayRedux2, self).setUp()
        self.ID = '247123464'

        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data_2, self.ID, self.PHASE_0)

        self.assertEqual(self.STATE_0[0], [1.19, 0.1])
        self.assertEqual(self.STATE_0[0], sol)

    def test_1(self):
        sol = process_delay(self.kernel_data_2, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1[0], [1.11, 0.03])
        self.assertEqual(self.STATE_1[0], sol)

    def test_rewards(self):
        reward = round(-0.01 * ((1.19 - 0.38 ) + (0.1 - 0.1 ) + (1.11 - 0.17) + (0.03 - 0.03)), 4)
        self.assertAlmostEqual(self.REWARD, reward)

class TestGridTLS3DelayRedux2(TestGridDelayReduxSetUp2):
    """
        * Tests maximize delay redux wrt Grid network

        * Set of tests that target the implemented
          problem formulations, i.e. state and reward
          function definitions.

        * Use lazy_properties to compute once and use
          as many times as you want -- it's a cached
          property
    """
    def setUp(self):
        """Code here will run before every test"""

        super(TestGridTLS3DelayRedux2, self).setUp()
        self.ID = '247123468'

        self.PHASE_0 = 0
        self.STATE_0 = self.state[self.ID][self.PHASE_0]

        self.PHASE_1 = 1
        self.STATE_1 = self.state[self.ID][self.PHASE_1]
        self.REWARD = self.reward(self.observation_space)[self.ID]

    def test_0(self):
        sol = process_delay(self.kernel_data_2, self.ID, self.PHASE_0)

        self.assertEqual(self.STATE_0[0], [0.06, 2.52])
        self.assertEqual(self.STATE_0[0], sol)

    def test_1(self):
        sol = process_delay(self.kernel_data_2, self.ID, self.PHASE_1)

        self.assertEqual(self.STATE_1[0], [0.01, 2.05])
        self.assertEqual(self.STATE_1[0], sol)

    def test_rewards(self):
        reward = round(-0.01 * ((1.19 - 0.38) + (0.1 - 0.1) + (1.11 - 0.17) + (0.03 - 0.03)), 4)

        reward = round(-0.01 * ((0.06 - 0.06) + (2.52 - 0.09) + (0.01 - 0.22) + (2.05 - 1.55)), 4)
        self.assertAlmostEqual(self.REWARD, reward)

def process_delay(kernel_data, node_id, phase_id, norm_vehs=False):
    """Processes batched delay computation"""
    cycle_time = 60

    def fn(x):
        return np.where(x >= 1, 0.0, np.exp(-5 * x))

    def green(x, y):
        if y == 0:
            return x[0].upper() in ('G', 'Y')
        else:
            return x[0].upper() not in ('G', 'Y')

    def ind(x, y):
        g = green(x, y)
        return int(g * 0 + (not g) * 1)


    delay_times = []
    weight = [0.0, 0.0]
    # timesteps = list(range(1, 60)) + [0]
    timesteps, vehs, tls = zip(*kernel_data)
    for t, vehicles, tls in zip(timesteps, vehs, tls):
        qt = defaultdict(lambda : [0.0, 0.0])
        vehs = vehicles[node_id][phase_id]
        tl = tls[node_id]
        index = ind(tl, phase_id)
        weight[index] += 1
        for veh in vehs:
            key = (veh.edge_id, veh.lane)
            qt[key][index] += fn(veh.speed / 13.89)

        if len(qt) == 0:
            delay_times.append([0.0, 0.0])
        else:
            if norm_vehs:
                delay_times.append([
                    sum([v / MAX_VEHS_PER_LANE[k]  for k, v in qt.items()]) for i in range(2)
                ])
            else:
                delay_times.append([sum([values[i] for values in qt.values()]) for i in range(2)])

    ret =  [round(sum([dt[i] for dt in delay_times]) / weight[i] , 2) for i in range(2)]
    return ret

if __name__ == '__main__':
    unittest.main()
