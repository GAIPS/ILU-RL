"""Objects that define the various meta-parameters of an experiment."""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-30'
import math
import numpy as np

import flow.core.params as flow_params

from collections import namedtuple
from ilurl.core.ql.reward import REWARD_TYPES
from ilurl.core.ql.choice import CHOICE_TYPES

from ilurl.dumpers.inflows import inflows_dump
from ilurl.loaders.nets import get_edges, get_routes, get_path
from ilurl.loaders.vtypes import get_vehicle_types

STATE_FEATURES = ('speed', 'count') #, 'flow', 'queue'

''' Bounds : namedtuple
        provide the settings to describe discrete variables ( e.g actions ). Or
        create discrete categorizations from continous variables ( e.g states)

    * rank: int
        Number of variable dimensions

    * depth: int
        Number of categories

'''
Bounds = namedtuple('Bounds', 'rank depth')

''' Reward : namedtuple
        Settings needed to perform reward computation

    * type: string
        A reward computation type in REWARD_TYPES

    * additional parameters: dict or None
        A dict containing additional parameters.

'''
Reward = namedtuple('Reward', 'type additional_params')

ADDITIONAL_PARAMS = {
    # every `switch` seconds concentrate flow in one direction
     "switch": 900
}

class MDPParams:
    """
        Holds general problem formulation params (MDP).
    """

    def __init__(self,
                num_actions,
                phases_per_traffic_light,
                states=('speed', 'count'),
                discretize_state_space=True,    # TODO
                normalize_state_space=True,     # TODO
                category_counts=[8.56, 13.00],  # TODO
                category_speeds=[2.28, 5.50],   # TODO
                reward={'type': 'target_velocity',
                        'additional_params': {
                            'target_velocity': 1.0
                        }
                }
            ):
        """Instantiate MDP params.

        PARAMETERS
        ----------
        * num_actions: dict
            a dictionary containing the number of actions per
            intersection. The outer keys are tls_ids.

        * phases_per_traffic_light: dict
            a dictionary containing the number of phases per 
            intersection. The outer keys are tls_ids.

        * states: ('speed', 'count')
            the features to be used as state space representation.

        * discretize_state_space: bool
            if True the state space will be categorized. TODO

        * normalize_state_space: bool
            if True the state space normalization will be applied. TODO

        * category_counts: TODO

        * category_speeds: TODO

        * reward: namedtuple (see Reward definition above)

        """
        kwargs = locals()

        for attr, value in kwargs.items():
            if attr not in ('self', 'states', 'rewards'):
                setattr(self, attr, value)

        # State space:
        if 'states' in kwargs:
            states_tuple = kwargs['states']
            for name in states_tuple:
                if name not in STATE_FEATURES:
                    raise ValueError(f'''
                        {name} must be in {STATE_FEATURES}
                    ''')
            self.states_labels = states_tuple

        # Reward function.
        reward = kwargs['reward']
        if reward['type'] not in REWARD_TYPES:
            raise ValueError(f'''
                Reward type must be in {REWARD_TYPES}. Got {reward['type']}
            type''')
        else:
            self.set_reward(reward['type'], reward['additional_params'])

        if self.normalize_state_space:
            if max(self.category_speeds) > 1:
                raise ValueError('If `normalize` flag is set categories'
                                    'must be between 0 and 1')

    def set_reward(self, type, additional_params):
        self.reward = Reward(type, additional_params)

    def categorize_space(self, observation_space):
        """Converts readings e.g averages, counts into integers

        PARAMETERS
        ----------
            * observation_space: a list of lists
                level 1 -- number of intersections controlled
                level 2 -- number of phases e.g 2
                level 3 -- number of variables

        RETURNS
        -------

        EXAMPLE
        -------
            # 1 tls, 2 phases, 2 variables
            > reading = [[[14.2, 3], [0, 10]]]
            > categories = categorize_space(reading)
            > categories
            > [[2, 0], [0, 3]]
        """

        labels = list(self.states_labels)

        categorized_space = {}
        # first loop is for intersections
        for tls_id in observation_space.keys():
            inters_space = observation_space[tls_id]
            # second loop is for phases
            categorized_intersection = []
            for phase_space in inters_space:
                # third loop is for variables
                categorized_phases = []
                for i, label in enumerate(labels):
                    val = phase_space[i]
                    category = getattr(self, f'_categorize_{label}')(val)
                    categorized_phases.append(category)
                categorized_intersection.append(categorized_phases)
            categorized_space[tls_id] = categorized_intersection
            
        
        return categorized_space

    def split_space(self, observation_space):
        """Splits different variables into tuple.
        
        PARAMETERS
        ----------
        * observation_space: list of lists
            nested 3 level list such that;
            The second level represents it's phases; e.g
            north-south and east-west. And the last level represents
            the variables withing labels e.g `speed` and `count`.

        RETURNS
        -------
            * flatten space

        EXAMPLE
        -------
        > observation_space = [[[13.3, 2.7], [15.7, 1.9]]]
        > splits = split_space(observation_space)
        > splits
        > [[13.3, 15.7], [2.7, 1.9]]

        """
        num_labels = len(self.states_labels)

        splits = []
        for label in range(num_labels):
            components = []
            for phases in observation_space:
                components.append(phases[label])
            splits.append(components)

        return splits

    def flatten_space(self, observation_space):
        """Linearizes hierarchial state representation.
        
        PARAMETERS
        ----------
            * observation_space: list of lists
            nested 2 level list such that;
            The second level represents it's phases; e.g
            north-south and east-west. And the last level represents
            the variables withing labels e.g `speed` and `count`.

        RETURNS
        -------
            * flattened_space: a list
            
        EXAMPLE
        -------
        > observation_space = [[[13.3, 2.7], [15.7, 1.9]]]
        > flattened = flatten_space(observation_space)
        > flattened
        > [13.3, 2.7, 15.7, 1.9]

        """
        out = {}

        for tls_id in observation_space.keys():
            tls_obs = observation_space[tls_id]

            flattened = [obs_value for phases in tls_obs
                        for obs_value in phases]

            out[tls_id] = tuple(flattened)

        return out

    def _categorize_speed(self, speed):
        """
            Converts a float speed into a category (integer).
        """
        return np.digitize(speed, bins=self.category_speeds).tolist()

    def _categorize_count(self, count):
        """
            Converts a float count into a category (integer).
        """
        return np.digitize(count, bins=self.category_counts).tolist()


class QLParams:
    """
        Base Q-learning parameters
    """

    def __init__(
            self,
            # Warning (alpha): a schedule is being used so
            # this parameter makes no difference.
            alpha=5e-1,
            # Warning (epsilon): a schedule is being used
            # so this parameter makes no difference.
            epsilon=3e-2,
            gamma=0.9,
            c=2,
            initial_value=0,
            choice_type='eps-greedy',
            replay_buffer=False,
            replay_buffer_size=500,
            replay_buffer_batch_size=64,
            replay_buffer_warm_up=200,
        ):
        """Instantiate Q-learning parameters.

        PARAMETERS
        ----------
        * alpha: float
            the learning rate the weight given to new knowledge [1].

        * epsilon: float
            the chance to adopt a random action instead of a greedy
            action [1].

        * gamma: float
            the discount rate [1].

        * c: int
            upper confidence bound (UCB) exploration constant.

        * choice_type: ('eps-greedy', 'optimistic', 'ucb')
            type of exploration strategy.

        * initial_value: float
            Q-table initialization value.

        * replay_buffer: bool
            if True batch learning will be used (Dyna-Q).

        * replay_buffer_size: int
            the size of the replay buffer.

        * replay_buffer_batch_size: int
            the size of the batches sampled from the replay buffer.

        * replay_buffer_warm_up: int
            replay buffer warm-up steps, i.e. the number of update
            steps before the learning starts.

        REFERENCES:
        ----------
            [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

        """
        kwargs = locals()

        if alpha <= 0 or alpha >= 1:
            raise ValueError('''The ineq 0 < alpha < 1 must hold.
                    Got alpha = {}.'''.format(alpha))

        if epsilon < 0 or epsilon > 1:
            raise ValueError('''The ineq 0 < epsilon < 1 must hold.
                    Got epsilon = {}.'''.format(epsilon))

        if gamma <= 0 or gamma > 1:
            raise ValueError('''The ineq 0 < gamma <= 1 must hold.
                    Got gamma = {}.'''.format(gamma))

        if choice_type not in CHOICE_TYPES:
            raise ValueError(
                f'''Choice type should be in {CHOICE_TYPES}
                    Got choice_type = {choice_type}.'''
            )

        if replay_buffer_size <= 0:
            raise ValueError('''The ineq replay_buffer_size > 0
                    must hold. Got replay_buffer_size = {}
                    '''.format(replay_buffer_size))

        if replay_buffer_batch_size <= 0 \
            or replay_buffer_batch_size > replay_buffer_size:
            raise ValueError('''The ineq replay_buffer_size >=
                    replay_buffer_batch_size > 0 must hold.
                    Got replay_buffer_batch_size = {}
                    '''.format(replay_buffer_batch_size))

        if replay_buffer_warm_up < 0:
            raise ValueError('''The ineq replay_buffer_warm_up
                    >= 0 must hold. Got replay_buffer_warm_up = {}
                    '''.format(replay_buffer_warm_up))

        for attr, value in kwargs.items():
            setattr(self, attr, value)


class InFlows(flow_params.InFlows):
    """InFlow: plus load & dump functionality"""

    @classmethod
    def make(cls, network_id, horizon, demand_type, label, initial_config=None):

        inflows = cls(network_id, horizon, demand_type,
                      initial_config=initial_config)
        # checks if route exists -- returning the path
        path = inflows_dump(
            network_id,
            inflows,
            distribution=demand_type,
            label=label
        )
        return path

    def __init__(self,
                 network_id,
                 horizon,
                 demand_type,
                 insertion_probability=0.1,
                 initial_config=None,
                 additional_params=ADDITIONAL_PARAMS):

        super(InFlows, self).__init__()

        if initial_config is not None:
            edges_distribution = initial_config.edges_distribution
        else:
            edges_distribution = None
        edges = get_edges(network_id)
        # an array of kwargs
        params = []
        for eid in get_routes(network_id):
            # use edges distribution to filter routes
            if ((edges_distribution is None) or
               (edges_distribution and eid in edges_distribution)):
                edge = [e for e in edges if e['id'] == eid][0]

                num_lanes = edge['numLanes'] if 'numLanes' in edge else 1

                args = (eid, 'human')
                if demand_type == 'lane':
                    kwargs = {
                        'probability': round(insertion_probability * num_lanes, 2),
                        'depart_lane': 'best',
                        'depart_speed': 'random',
                        'name': f'lane_{eid}',
                        'begin': 1,
                        'end': horizon
                    }

                    params.append((args, kwargs))
                elif demand_type == 'switch':
                    switch = additional_params['switch']
                    num_flows = max(math.ceil(horizon / switch), 1)
                    for hr in range(num_flows):
                        step = min(horizon - hr * switch, switch)
                        # switches in accordance to the number of lanes
                        if (hr + num_lanes) % 2 == 1:
                            insertion_probability = insertion_probability \
                                                    + 0.2 * num_lanes

                        kwargs = {
                            'probability': round(insertion_probability, 2),
                            'depart_lane': 'best',
                            'depart_speed': 'random',
                            'name': f'switch_{eid}',
                            'begin': 1 + hr * switch,
                            'end': step + hr * switch
                        }

                        params.append((args, kwargs))
                else:
                    raise ValueError(f'Unknown demand_type {demand_type}')

        # Sort params flows will be consecutive
        params = sorted(params, key=lambda x: x[1]['end'])
        params = sorted(params, key=lambda x: x[1]['begin'])
        for args, kwargs in params:
            self.add(*args, **kwargs)

class NetParams(flow_params.NetParams):
    """Extends NetParams to work with saved templates"""

    @classmethod
    def from_template(cls, network_id, horizon, demand_type,
                      label=None, initial_config=None):
        """Factory method based on {network_id} layout + configs

        Params:
        -------
        *   network_id: string
            standard {network_id}.net.xml file, ex: `intersection`
            see data/networks for a list
        *   horizon: integer
            latest depart time
        *   demand_type: string
            string
        *   label: string
            e.g `eval, `train` or `test`

        Returns:
        -------
        *   ilurl.core.params.NetParams
            network parameters SEE parent
        """
        net_path = get_path(network_id, 'net')
        # TODO: test if exists first!
        rou_path = InFlows.make(network_id, horizon,
                                demand_type, label=label,
                                initial_config=initial_config)
        vtype_path = get_vehicle_types()
        return cls(
            template={
                'net': net_path,
                'vtype': vtype_path,
                'rou': rou_path
            }
        )

    @classmethod
    def load(cls, network_id, route_path):
        """Loads paremeters from net {network_id} and
            routes from {route_path}

        Params:
        -------
        *   network_id: string
            standard {network_id}.net.xml file, ex: `intersection`
            see data/networks for a list
        *   route_path: string
            valid path on disk for a *.rou.xml file

        Returns:
        -------
        *   ilurl.core.params.NetParams
            network parameters SEE parent
        """
        net_path = get_path(network_id, 'net')
        vtype_path = get_vehicle_types()

        return cls(
            template={
                'net': net_path,
                'vtype': vtype_path,
                'rou': [route_path]
            }
        )
