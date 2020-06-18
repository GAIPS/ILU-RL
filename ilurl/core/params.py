"""Objects that define the various meta-parameters of an experiment."""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-30'
import math
import numpy as np

import flow.core.params as flow_params

from collections import namedtuple
from ilurl.core.rewards import get_rewards 
from ilurl.agents.ql.choice import CHOICE_TYPES

from ilurl.dumpers.inflows import inflows_dump
from ilurl.loaders.nets import get_edges, get_routes, get_path
from ilurl.loaders.vtypes import get_vehicle_types
from ilurl.loaders.demands import get_demand

STATE_FEATURES = ('speed', 'count', 'delay') #, 'flow', 'queue'

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

TLS_TYPES = ('controlled', 'actuated', 'static', 'random')

DEMAND_TYPES = ('constant', 'variable') # TODO: Add switch demand type.


class Printable(object):
    def __repr__(self):
        """Returns a string containing the attributes of the class."""
        text_repr = f"\n{self.__class__.__name__}:\n"
        for (attr, val) in self.__dict__.items():
            text_repr += f"{attr}: {val}\n"
        return text_repr


class MDPParams(Printable):
    """
        Holds general problem formulation params (MDP).
    """

    def __init__(self,
                states=('speed', 'count'),
                discretize_state_space=True,    # TODO
                normalize_state_space=True,     # TODO
                category_counts=[8.56, 13.00],
                category_delays=[5, 30],
                category_speeds=[2.28, 5.50],
                reward = 'MaxSpeedCountReward',
                reward_rescale=1.0,
                target_velocity=1.0,
                velocity_threshold=None, 
            ):
        """Instantiate MDP params.

        Parameters:
        ----------
        * states: ('speed', 'count')
            the features to be used as state space representation.

        * discretize_state_space: bool
            if True the state space will be categorized. TODO

        * normalize_state_space: bool
            if True the state space normalization will be applied. TODO

        * category_counts: TODO

        * category_speeds: TODO

        * category_delays: TODO

        * reward: namedtuple (see Reward definition above)

        * reward_rescale: float
            Reward rescaling factor.

        """
        kwargs = locals()

        for attr, value in kwargs.items():
            if attr not in ('self'):
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

        if self.normalize_state_space:
            if max(self.category_speeds) > 1:
                raise ValueError('If `normalize` flag is set categories'
                                    'must be between 0 and 1')

    # def categorize_space(self, observation_space):
    #     """Converts readings e.g averages, counts into integers

    #     Parameters:
    #     ----------
    #         * observation_space: a list of lists
    #             level 1 -- number of intersections controlled
    #             level 2 -- number of phases e.g 2
    #             level 3 -- number of variables

    #     Returns:
    #     -------

    #     Example:
    #     -------
    #         # 1 tls, 2 phases, 2 variables
    #         > reading = [[[14.2, 3], [0, 10]]]
    #         > categories = categorize_space(reading)
    #         > categories
    #         > [[2, 0], [0, 3]]
    #     """

    #     labels = list(self.states_labels)

    #     categorized_space = {}
    #     # first loop is for intersections
    #     for tls_id in observation_space.keys():
    #         inters_space = observation_space[tls_id]
    #         # second loop is for phases
    #         categorized_intersection = []
    #         for phase_space in inters_space:
    #             # third loop is for variables
    #             categorized_phases = []
    #             for i, label in enumerate(labels):
    #                 val = phase_space[i]
    #                 category = getattr(self, f'_categorize_{label}')(val)
    #                 categorized_phases.append(category)
    #             categorized_intersection.append(categorized_phases)
    #         categorized_space[tls_id] = categorized_intersection
    #         
    #     
    #     return categorized_space

    # def split_space(self, observation_space):
    #     """Splits different variables into tuple.
    #     
    #     Parameters:
    #     ----------
    #     * observation_space: list of lists
    #         nested 3 level list such that;
    #         The second level represents it's phases; e.g
    #         north-south and east-west. And the last level represents
    #         the variables withing labels e.g `speed` and `count`.

    #     Returns:
    #     -------
    #         * flatten space

    #     Example:
    #     -------
    #     > observation_space = [[[13.3, 2.7], [15.7, 1.9]]]
    #     > splits = split_space(observation_space)
    #     > splits
    #     > [[13.3, 15.7], [2.7, 1.9]]

    #     """
    #     num_labels = len(self.states_labels)

    #     splits = []
    #     for label in range(num_labels):
    #         components = []
    #         for phases in observation_space:
    #             components.append(phases[label])
    #         splits.append(components)

    #     return splits

    # def flatten_space(self, observation_space):
    #     """Linearizes hierarchial state representation.
    #     
    #     Parameters:
    #     ----------
    #         * observation_space: list of lists
    #         nested 2 level list such that;
    #         The second level represents it's phases; e.g
    #         north-south and east-west. And the last level represents
    #         the variables withing labels e.g `speed` and `count`.

    #     Returns:
    #     -------
    #         * flattened_space: a list
    #         
    #     Example:
    #     -------
    #     > observation_space = [[[13.3, 2.7], [15.7, 1.9]]]
    #     > flattened = flatten_space(observation_space)
    #     > flattened
    #     > [13.3, 2.7, 15.7, 1.9]

    #     """
    #     out = {}

    #     for tls_id in observation_space.keys():
    #         tls_obs = observation_space[tls_id]

    #         flattened = [obs_value for phases in tls_obs
    #                     for obs_value in phases]

    #         out[tls_id] = tuple(flattened)

    #     return out

    # def _categorize_speed(self, speed):
    #     """
    #         Converts a float speed into a category (integer).
    #     """
    #     return np.digitize(speed, bins=self.category_speeds).tolist()

    # def _categorize_count(self, count):
    #     """
    #         Converts a float count into a category (integer).
    #     """
    #     return np.digitize(count, bins=self.category_counts).tolist()


class QLParams(Printable):
    """
        Base Q-learning parameters.
    """

    def __init__(
            self,
            lr_decay_power_coef=0.66,
            eps_decay_power_coef=1.0,
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

        Parameters:
        ----------
        * lr_decay_power_coef: float
            the learning rate decay power coefficient value, i.e. the 
            learning rate is calculated using the following expression:
                Power schedule (input x): 1 / ((1 + x)**lr_decay_power_coef)

        * eps_decay_power_coef: float
            the epsilon decay decay power coefficient value, i.e. the
            epsilon (chance do take a random action) is calculated
            using the following expression:
                Power schedule (input x): 1 / ((1 + x)**eps_decay_power_coef)

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
            replay buffer warm-up steps, i.e. the number of
            steps before the learning starts.

        References:
        ----------
            [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

        """
        kwargs = locals()

        if lr_decay_power_coef <= 0:
            raise ValueError('''The ineq 0 < lr_decay_power_coef must hold.
                    Got lr_decay_power_coef = {}.'''.format(
                        lr_decay_power_coef))

        if eps_decay_power_coef <= 0:
            raise ValueError('''The ineq 0 < eps_decay_power_coef.
                    Got eps_decay_power_coef = {}.'''.format(
                        eps_decay_power_coef))

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
            if attr not in ('self'):
                setattr(self, attr, value)


class DQNParams(Printable):
    """
        Base Deep Q-network parameters.
    """

    def __init__(
            self,
            learning_rate=1e-3,
            gamma=0.9,
            batch_size=256,
            prefetch_size=4,
            target_update_period=100,
            samples_per_insert=32.0,
            min_replay_size=1000,
            max_replay_size=1000000,
            importance_sampling_exponent=0.2,
            priority_exponent=0.6,
            n_step=5,
            epsilon=0.05,
        ):
        """Instantiate Deep Q-network parameters.

        Parameters:
        ----------
        * learning_rate: float
            learning rate.

        * gamma: float
            the discount factor.

        * batch_size: int
            the size of the batches sampled from the replay buffer.

        * prefetch_size: int

        * target_update_period: int
            target network updates interval.

        * samples_per_insert: float
            number of samples to take from replay for every insert
            that is made.
        
        * min_replay_size: int
            minimum replay size before updating. This and all
            following arguments are related to dataset construction and will be
            ignored if a dataset argument is passed.

        * max_replay_size: int
            maximum replay size.

        * importance_sampling_exponent: float
            power to which importance weights are raised
            before normalizing.

        * priority_exponent: float
            exponent used in prioritized sampling.

        * n_step: int
            number of steps to squash into a single transition.

        * epsilon: float
            probability of taking a random action; ignored if a policy
            network is given.

        """
        kwargs = locals()

        if learning_rate <= 0 or learning_rate >= 1:
            raise ValueError('''The ineq 0 < lr < 1 must hold.
                    Got lr = {}.'''.format(learning_rate))

        if gamma <= 0 or gamma > 1:
            raise ValueError('''The ineq 0 < gamma <= 1 must hold.
                    Got gamma = {}.'''.format(gamma))

        if epsilon <= 0 or epsilon > 1:
            raise ValueError('''The ineq 0 < epsilon <= 1 must hold.
                    Got epsilon = {}.'''.format(epsilon))

        if samples_per_insert <= 0:
            raise ValueError('''The ineq samples_per_insert > 0
                    must hold. Got samples_per_insert = {}
                    '''.format(samples_per_insert))

        if max_replay_size <= 0:
            raise ValueError('''The ineq max_replay_size > 0
                    must hold. Got max_replay_size = {}
                    '''.format(max_replay_size))

        if batch_size <= 0 \
            or batch_size > max_replay_size:
            raise ValueError('''The ineq max_replay_size >=
                    batch_size > 0 must hold.
                    Got batch_size = {}
                    '''.format(batch_size))

        if min_replay_size < 0:
            raise ValueError('''The ineq 0 <= min_replay_size.
                    Got min_replay_size = {}.'''.format(min_replay_size))

        if target_update_period <= 0:
            raise ValueError('''The ineq 0 < target_net_update_interval.
                    Got target_net_update_interval = {}.'''.format(
                        target_update_period))

        if n_step <= 0:
            raise ValueError('''The ineq 0 < n_step.
                    Got n_step = {}.'''.format(n_step))

        for attr, value in kwargs.items():
            if attr not in ('self'):
                setattr(self, attr, value)


class TrainParams(Printable):
    """
        Base train.py parameters.
    """

    def __init__(
            self,
            network='intersection',
            experiment_time=900000,
            experiment_save_agent=False,
            experiment_save_agent_interval=2500,
            experiment_seed=None,
            sumo_render=False,
            sumo_emission=False,
            tls_type='controlled',
            demand_type='constant',
        ):
        """Instantiate train parameters.

        Parameters:
        ----------
        * network: str
            Network to be simulated.

        * experiment_time: int
            Simulation's real world time in seconds.

        * experiment_save_agent: bool
            Whether to save RL-agent parameters (checkpoints)
            throughout training.

        * experiment_save_agent_interval: int
            [Only applies if experiment_save_agent is True]
            Save agent interval (in agent update steps).

        * experiment_seed: int or None
            Sets seed value for both RL agent and SUMO.
            `None` for rl agent defaults to RandomState()
            `None` for Sumo defaults to a fixed but arbitrary seed.

        * sumo_render: bool
            If True renders the simulation.

        * sumo_emission: bool
            If True saves emission data from simulation on /data/emissions.

        * tls_type: ('controlled', 'static', 'random' or 'actuated')
            SUMO traffic light type: \'controlled\', \'actuated'\',
                    \'static\' or \'random\'.

        * demand_type: ('constant' or 'variable')
            constant - uniform vehicles demand.
            variable - realistic demand that varies throught 24 hours
                    (resembling realistic traffic variations, e.g. peak hours)

        """
        kwargs = locals()

        if experiment_time <= 0:
            raise ValueError('''The ineq 0 < experiment_time must hold.
                    Got experiment_time = {}.'''.format(experiment_time))

        if experiment_save_agent_interval <= 0:
            raise ValueError('''The ineq 0 < experiment_save_agent_interval < 1 must hold.
                    Got experiment_save_agent_interval = {}.'''.format(experiment_save_agent_interval))

        if tls_type not in TLS_TYPES:
            raise ValueError('''The tls_type must be in ('controlled', 'static', 'random', 'actuated').
                    Got tls_type = {}.'''.format(tls_type))

        if demand_type not in DEMAND_TYPES:
            raise ValueError('''The demand_type must be in ('constant' or 'variable').
                    Got demand_type = {}.'''.format(demand_type))

        for attr, value in kwargs.items():
            if attr not in ('self'):
                setattr(self, attr, value)


class InFlows(flow_params.InFlows,Printable):
    """InFlow: plus load & dump functionality"""

    @classmethod
    def make(cls, network_id, horizon,
            demand_type, label, initial_config=None):

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
                 initial_config=None):

        super(InFlows, self).__init__()

        if initial_config is not None:
            edges_distribution = initial_config.edges_distribution
        else:
            edges_distribution = None
        edges = get_edges(network_id)

        # Get demand data.
        demand = get_demand(demand_type, network_id)

        # an array of kwargs
        params = []
        for eid in get_routes(network_id):
            # use edges distribution to filter routes
            if ((edges_distribution is None) or
               (edges_distribution and eid in edges_distribution)):
                edge = [e for e in edges if e['id'] == eid][0]

                num_lanes = edge['numLanes'] if 'numLanes' in edge else 1

                args = (eid, 'human')

                # Uniform flows.
                if demand_type == 'constant':

                    insertion_probability = demand[str(num_lanes)]

                    kwargs = {
                        'probability': insertion_probability,
                        'depart_lane': 'best',
                        'depart_speed': 'random',
                        'name': f'uniform_{eid}',
                        'begin': 1,
                        'end': horizon
                    }

                    params.append((args, kwargs))

                # 24 hours variable flows.
                elif demand_type == 'variable':

                    peak_distribution = demand['peak']

                    num_days = horizon // (24*3600) + 1

                    for day in range(num_days):

                        for hour in range(24):

                            hour_factor = demand['hours'][str(hour)]

                            insertion_probability = hour_factor * \
                                        peak_distribution[str(num_lanes)]

                            begin_t = 1 + (hour * 3600) + (24*3600) * day
                            end_t = (hour * 3600) + (24*3600) * day + 3600

                            # print('-'*10)
                            # print('num_lanes:', num_lanes)
                            # print('begin_t:', begin_t)
                            # print('end_t:', end_t)
                            # print('insertion_probability:', insertion_probability)
                            # print('\n')

                            if end_t > horizon:
                                break

                            kwargs = {
                                'probability': insertion_probability,
                                'depart_lane': 'best',
                                'depart_speed': 'random',
                                'name': f'variable_{eid}',
                                'begin': begin_t,
                                'end': end_t
                            }

                            params.append((args, kwargs))

                elif demand_type == 'switch':
                    raise NotImplementedError('Switch demand')

                    """ switch = additional_params['switch']
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

                        params.append((args, kwargs)) """
                else:
                    raise ValueError(f'Unknown demand_type {demand_type}')

        # Sort params flows will be consecutive
        params = sorted(params, key=lambda x: x[1]['end'])
        params = sorted(params, key=lambda x: x[1]['begin'])
        for args, kwargs in params:
            self.add(*args, **kwargs)


class NetParams(flow_params.NetParams,Printable):
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
