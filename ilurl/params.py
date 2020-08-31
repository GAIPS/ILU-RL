"""Objects that define the various meta-parameters of an experiment."""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-30'
import math
from typing import List, Tuple
from collections import namedtuple

import numpy as np

import flow.core.params as flow_params

from ilurl.rewards import get_rewards
from ilurl.agents.ql.choice import CHOICE_TYPES
from ilurl.loaders.nets import get_edges, get_routes, get_path
from ilurl.loaders.vtypes import get_vehicle_types
from ilurl.loaders.demands import get_demand
from ilurl.utils.aux import Printable


''' Bounds : namedtuple
        Provide the settings to describe discrete variables ( e.g actions ). Or
        create discrete categorizations from continous variables ( e.g states).

    * rank: int
        Number of variable dimensions.

    * depth: int
        Number of categories.
'''
Bounds = namedtuple('Bounds', 'rank depth')

# State space features.
STATE_FEATURES = ('speed', 'count', 'delay', 'queue') #, 'flow'

# Traffic light system types.
TLS_TYPES = ('rl', 'static', 'webster',
             'actuated', 'random', 'max_pressure')

# Traffic demand types (flows).
DEMAND_TYPES = ('constant', 'variable') # TODO: Add 'switch' demand type.


class MDPParams(Printable):
    """
        Holds general problem formulation parameters (MDP).
    """

    def __init__(self,
                discount_factor: float = 0.95,
                action_space: str = 'discrete',
                features: Tuple[str] = ('speed', 'count'),
                normalize_state_space: bool = True,
                discretize_state_space: bool = True,
                category_counts: List[float] = [8.56, 13.00],
                category_speeds: List[float] = [2.28, 5.50],
                category_delays: List[float] = [5, 30],
                category_queues: List[float] = [1, 10],
                category_times: List[int] = [1, 10],
                category_pressures: List[int] = [1, 10],
                category_average_pressures: List[int] = [1, 10],
                category_flows: List[int] = [1, 10],
                category_speed_scores: List[int] = [1, 10],
                reward: str = 'reward_max_speed_count',
                reward_rescale: float = 1.0,
                time_period: int = None,
                velocity_threshold = None,
            ):
        """Instantiate MDP params.

        Parameters:
        ----------

        * discount_factor: float
            MDP discount factor (gamma)

        * action_space: ('discrete' or 'continuous')
            Whether the action space is continuous (cycle length allocation)
            or discrete (choose from a set of signal plans).

        * states: ('speed', 'count', ...)
            the features to be used as state space representation.

        * normalize_state_space: bool
            if True the state space normalization will be applied.

        * discretize_state_space: bool
            if True the state space will be categorized (categories below).

        * category_counts: List[float]
        * category_speeds: List[float]
        * category_delays: List[float]
        * category_queues: List[float]

        * reward: str
            The reward function to be applied.

        * reward_rescale: float
            Reward rescaling factor.

        * velocity_threshold: float
            Additional parameter used in the reward computation.

        """
        kwargs = locals()

        for attr, value in kwargs.items():
            if attr not in ('self'):
                setattr(self, attr, value)

        # State space.
        if 'states' in kwargs:
            self.states_labels = kwargs['states']

        if self.normalize_state_space:
            if max(self.category_speeds) > 1:
                raise ValueError('If `normalize` flag is set categories'
                                    'must be between 0 and 1.')

        # Action space.
        if self.action_space not in ('discrete', 'continuous'):
            raise ValueError('Action space must be either \'discrete\' or \'continuous\'')

        # Discount factor.
        if not (0 < self.discount_factor < 1):
            raise ValueError('Discount factor must be between 0 and 1.')


class QLParams(Printable):
    """
        Base Q-learning parameters.
    """

    def __init__(
            self,
            lr_decay_power_coef: float = 0.66,
            eps_decay_power_coef: float = 1.0,
            c: int = 2,
            initial_value: float = 0,
            choice_type: str = 'eps-greedy',
            replay_buffer: bool = False,
            replay_buffer_size: int = 500,
            replay_buffer_batch_size: int = 64,
            replay_buffer_warm_up: int = 200,
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
            learning_rate: float = 1e-3,
            batch_size: int = 256,
            prefetch_size: int = 1,
            target_update_period: int = 100,
            samples_per_insert: float = 32.0,
            min_replay_size: int = 1000,
            max_replay_size: int = 1000000,
            importance_sampling_exponent: float = 0.2,
            priority_exponent: float = 0.6,
            n_step: int = 5,
            epsilon_init: float = 1.0,
            epsilon_final: float = 0.01,
            epsilon_schedule_timesteps: int = 50000,
            torso_layers : list = [5],
            head_layers  : list = [5],
        ):
        """Instantiate Deep Q-network parameters.

        Parameters:
        ----------
        * (See acme.agents.tf.dqn.agent.py file for more info).

        * learning_rate: float
            learning rate.

        * batch_size: int
            the size of the batches sampled from the replay buffer.

        * prefetch_size: int
            The number of batches to prefetch in the data tf pipeline.

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

        * epsilon_init: float
            Initial epsilon value (probability of taking a random action).

        * epsilon_final: float
            Final epsilon value (probability of taking a random action).

        * epsilon_schedule_timesteps: int
            Number of timesteps to decay epsilon from 'epsilon_init' to 'epsilon_final'.

        * torso_layers: list
            Torso MLP network layers.

        * head_layers: list
            Head (duelling) MLP network layers.

        """
        kwargs = locals()

        if learning_rate <= 0 or learning_rate >= 1:
            raise ValueError('''The ineq 0 < lr < 1 must hold.
                    Got lr = {}.'''.format(learning_rate))

        if epsilon_init < 0 or epsilon_init > 1:
            raise ValueError('''The ineq 0 < epsilon_init <= 1 must hold.
                    Got epsilon_init = {}.'''.format(epsilon))

        if epsilon_final < 0 or epsilon_final > 1:
            raise ValueError('''The ineq 0 < epsilon_final <= 1 must hold.
                    Got epsilon_final = {}.'''.format(epsilon))

        if epsilon_schedule_timesteps <= 0:
            raise ValueError('''The ineq epsilon_schedule_timesteps > 0
                    must hold. Got epsilon_schedule_timesteps = {}
                    '''.format(samples_per_insert))

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


class R2D2Params(Printable):
    """
        Base R2D2 parameters.
    """

    def __init__(
            self,
            burn_in_length: int,
            trace_length: int,
            replay_period: int,
            batch_size: int,
            prefetch_size: int,
            target_update_period: int,
            importance_sampling_exponent: float,
            priority_exponent: float,
            learning_rate: float,
            min_replay_size: int,
            max_replay_size: int,
            samples_per_insert: float,
            store_lstm_state: bool,
            max_priority_weight: float,
            epsilon_init: float,
            epsilon_final: float,
            epsilon_schedule_timesteps: int,
            rnn_hidden_size: int,
            head_layers: list
        ):
        """ Instantiate R2D2 parameters.

        Parameters:
        ----------
        * (See acme.agents.tf.r2d2.agent.py file for more info).

        * epsilon_init: float
            Initial epsilon value (probability of taking a random action).

        * epsilon_final: float
            Final epsilon value (probability of taking a random action).

        * epsilon_schedule_timesteps: int
            Number of timesteps to decay epsilon from 'epsilon_init' to 'epsilon_final'.

        * rnn_hidden_size: int
            Number of nodes in the RNN core of the network.

        * head_layers: list
            Head (duelling) MLP network layers.

        """
        kwargs = locals()

        # TODO: Add arguments restrictions.

        for attr, value in kwargs.items():
            if attr not in ('self'):
                setattr(self, attr, value)


class DDPGParams(Printable):
    """
        Base DDPG parameters.
    """

    def __init__(
            self,
            batch_size: int = 100,
            prefetch_size: int = 1,
            target_update_period: int = 100,
            min_replay_size: int = 1000,
            max_replay_size: int = 30000,
            samples_per_insert: float = 50.0,
            n_step: int = 5,
            sigma_init: float = 0.3,
            sigma_final: float = 0.01,
            sigma_schedule_timesteps: float = 45000,
            clipping: bool = True,
            policy_layers: list = [5, 5],
            critic_layers: list = [5, 5],
        ):
        """Instantiate DDPG parameters.

        Parameters:
        ----------
        * (See acme.agents.tf.ddpg.agent.py file for more info).

         """
        kwargs = locals()

        # TODO: Add arguments restrictions.

        for attr, value in kwargs.items():
            if attr not in ('self'):
                setattr(self, attr, value)


class TrainParams(Printable):
    """
        Base train.py parameters.
    """

    def __init__(
            self,
            network: str = 'intersection',
            experiment_time: int = 900000,
            experiment_save_agent: bool = False,
            experiment_save_agent_interval: int = 2500,
            experiment_seed=None,
            sumo_render: bool = False,
            sumo_emission: bool = False,
            tls_type: str = 'rl',
            demand_type: str = 'constant',
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

        * tls_type: ('rl', 'webster', 'static', 'random', 'actuated' or 'max-pressure')

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
            raise ValueError('''The tls_type must be in ('rl', 'webster', 'static', 'random', 'actuated' or 'max-pressure').
                    Got tls_type = {}.'''.format(tls_type))

        if demand_type not in DEMAND_TYPES:
            raise ValueError('''The demand_type must be in ('constant' or 'variable').
                    Got demand_type = {}.'''.format(demand_type))

        for attr, value in kwargs.items():
            if attr not in ('self'):
                setattr(self, attr, value)


class InFlows(flow_params.InFlows,Printable):
    """
        InFlows.
    """

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
