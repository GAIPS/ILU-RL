"""Objects that define the various meta-parameters of an experiment."""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-30'
import math
import numpy as np

import flow.core.params as flow_params

from collections import namedtuple
from ilurl.core.rewards import get_rewards 
from ilurl.core.ql.choice import CHOICE_TYPES

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


class MDPParams:
    """
        Holds general problem formulation params (MDP).
    """

    def __init__(self,
                states=('speed', 'count'),
                discretize_state_space=True,    # TODO
                normalize_state_space=True,     # TODO
                category_counts=[8.56, 13.00],  # TODO
                category_speeds=[2.28, 5.50],   # TODO
                reward = 'MaxSpeedCountReward',
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

        * reward: namedtuple (see Reward definition above)

        """
        kwargs = locals()

        for attr, value in kwargs.items():
            if attr not in ('self', 'states'):
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


class QLParams:
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
            setattr(self, attr, value)


class DQNParams:
    """
        Base Deep Q-network parameters.
    """

    def __init__(
            self,
            lr=5e-4,
            gamma=0.8,
            buffer_size=20000,
            batch_size=64,
            prioritized_replay=False,
            prioritized_replay_alpha=0.6,
            prioritized_replay_beta0=0.4,
            prioritized_replay_beta_iters=40000,
            prioritized_replay_eps=1e-6,
            exp_initial_p=1.0,
            exp_final_p=0.02,
            exp_schedule_timesteps=40000,
            learning_starts=2000,
            target_net_update_interval=2000,
            network_type='mlp',
            head_network_mlp_hiddens=[],
            head_network_layer_norm=False,
            head_network_dueling=False,
            mlp_hiddens=[8],
            mlp_layer_norm=False,
        ):
        """Instantiate Deep Q-network parameters.

        Parameters:
        ----------
        * lr: float
            learning rate.

        * gamma: float
            the discount rate.

        * buffer_size: int
            the size of the replay buffer.

        * batch_size: int
            the size of the batches sampled from the replay buffer.

        * prioritized_replay: bool
            Whether to use prioritized replay buffer.
            REF: https://arxiv.org/pdf/1511.05952.pdf

        * prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer

        * prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer

        * prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0.

        * prioritized_replay_eps: float
            epsilon to add to the TD errors when updating priorities.

        * exp_initial_p: float
            initial exploration rate.

        * exp_final_p: float
            final exploration rate.

        * exp_schedule_timesteps: int
            exploration decay interval i.e. the number of steps
            it takes to decay exp_initial_p to exp_final_p.

        * learning_starts: int
            the number of steps before the learning starts.

        * target_net_update_interval: int
            target network updates interval.

        * network_type: str
            See baselines.common.models for available models.

        * head_network_mlp_hiddens: list
            List with the hidden nodes (Q-network head).

        * head_network_layer_norm: bool
            Layer normalization (Q-network head).

        * head_network_dueling: bool,
            Dueling network.
            REF: http://proceedings.mlr.press/v48/wangf16.pdf

        * mlp_hiddens: list
            List with the hidden nodes (Applies if network_type= 'mlp').

        * mlp_layer_norm: bool
            Layer normalization (Applies if network_type= 'mlp').

        """
        kwargs = locals()

        if lr <= 0 or lr >= 1:
            raise ValueError('''The ineq 0 < lr < 1 must hold.
                    Got lr = {}.'''.format(lr))

        if gamma <= 0 or gamma > 1:
            raise ValueError('''The ineq 0 < gamma <= 1 must hold.
                    Got gamma = {}.'''.format(gamma))

        if exp_initial_p < 0 or exp_initial_p > 1:
            raise ValueError('''The ineq 0 < exp_initial_p < 1 must hold.
                    Got exp_initial_p = {}.'''.format(exp_initial_p))

        if exp_final_p < 0 or exp_final_p > 1:
            raise ValueError('''The ineq 0 < exp_final_p < 1 must hold.
                    Got exp_final_p = {}.'''.format(exp_final_p))

        if exp_schedule_timesteps < 0:
            raise ValueError('''The ineq 0 < exp_schedule_timesteps.
                    Got exp_schedule_timesteps = {}.'''.format(
                        exp_schedule_timesteps))

        if buffer_size <= 0:
            raise ValueError('''The ineq buffer_size > 0
                    must hold. Got buffer_size = {}
                    '''.format(buffer_size))

        if batch_size <= 0 \
            or batch_size > buffer_size:
            raise ValueError('''The ineq buffer_size >=
                    batch_size > 0 must hold.
                    Got batch_size = {}
                    '''.format(batch_size))

        if learning_starts < 0:
            raise ValueError('''The ineq 0 <= learning_starts.
                    Got learning_starts = {}.'''.format(learning_starts))

        if target_net_update_interval <= 0:
            raise ValueError('''The ineq 0 < target_net_update_interval.
                    Got target_net_update_interval = {}.'''.format(
                        target_net_update_interval))

        for attr, value in kwargs.items():
            setattr(self, attr, value)


class TrainParams:
    """
        Base train.py parameters.
    """

    def __init__(
            self,
            network='intersection',
            experiment_time=900000,
            experiment_log=False,
            experiment_log_interval=1000,
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

        * experiment_log: bool
            Whether to save experiment-related data in a JSON file
            thoughout training (allowing to live track training).
            If True tensorboard logs will also be created.

        * experiment_log_interval: int
            [Only applies if experiment_log is True]
            Log into JSON file interval (in agent update steps).

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

        if experiment_log_interval <= 0:
            raise ValueError('''The ineq 0 < experiment_log_interval must hold.
                    Got experiment_log_interval = {}.'''.format(experiment_log_interval))

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
            setattr(self, attr, value)


class InFlows(flow_params.InFlows):
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
