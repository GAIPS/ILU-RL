import os
import random
from typing import Sequence

import numpy as np

import tensorflow as tf
import sonnet as snt

import dm_env
import acme
from acme import specs
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils

from ilurl.agents.ddpg import acme_agent
from ilurl.agents.worker import AgentWorker
from ilurl.interfaces.agents import AgentInterface
from ilurl.utils.default_logger import make_default_logger
from ilurl.utils.precision import double_to_single_precision
from ilurl.utils.tf2_layers import InputStandardization


_TF_USE_GPU = False
_TF_NUM_THREADS = 1


class PolicyMLP(snt.Module):

    def __init__(self,
                hidden_layer_sizes: Sequence[int],
                actions_dim : int):
        """ Policy network.

        Args:
        hidden_layer_sizes: a sequence of ints specifying the size of each layer.
        action dim: actions number of dimensions.

        """
        super().__init__(name='layer_input_norm_mlp')

        layers = []

        # Hidden layers.
        for layer_size in hidden_layer_sizes:
            layers.append(snt.Linear(layer_size, w_init=tf.initializers.VarianceScaling(
                                        distribution='uniform', mode='fan_out', scale=0.333)))
            # layers.append(snt.LayerNorm(axis=slice(1, None), create_scale=True, create_offset=True))
            layers.append(tf.nn.relu)

        # Last layer.
        layers.append(networks.NearZeroInitializedLinear(actions_dim))
        layers.append(tf.nn.softmax)

        self._network = snt.Sequential(layers)

    def __call__(self, observations: types.Nest) -> tf.Tensor:
        """Forwards the policy network."""
        return self._network(tf2_utils.batch_concat(observations))


def _make_networks(
        actions_dim : int,
        state_dim : int,
        policy_layers : Sequence[int] = [5, 5],
        critic_layers : Sequence[int] = [5, 5],
    ):

    # Create the policy network.
    policy_network = PolicyMLP(policy_layers, actions_dim)

    # Create the critic network.
    critic_layers = list(critic_layers) + [1]
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        snt.nets.MLP(critic_layers, activate_final=False),
    ])

    observation_network = InputStandardization(shape=state_dim)

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }


class DDPG(AgentWorker,AgentInterface):
    """
        DDPG agent.
    """
    def __init__(self, *args, **kwargs):
        super(DDPG, self).__init__(*args, **kwargs)

    def init(self, params):

        if not _TF_USE_GPU:
            tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(_TF_NUM_THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(_TF_NUM_THREADS)

        if params.seed:
            agent_seed = params.seed + sum([ord(c) for c in params.name])
            random.seed(agent_seed)
            np.random.seed(agent_seed)
            tf.random.set_seed(agent_seed)

        # Internalize params.
        self._params = params

        self._name = params.name

        # Whether learning stopped.
        self._stop = False

        # Define specs. Everything needs to be single precision by default.
        observation_spec = specs.Array(shape=(params.states.rank,),
                                  dtype=np.float32,
                                  name='obs'
        )
        action_spec = specs.BoundedArray(shape=(params.num_phases,),
                                        dtype=np.float32,
                                        minimum=0.,
                                        maximum=1.,
                                        name='action'
        )
        reward_spec = specs.Array(shape=(),
                                  dtype=np.float32,
                                  name='reward'
        )
        discount_spec = specs.BoundedArray(shape=(),
                                           dtype=np.float32,
                                           minimum=0.,
                                           maximum=1.,
                                           name='discount'
        )

        env_spec = specs.EnvironmentSpec(observations=observation_spec,
                                          actions=action_spec,
                                          rewards=reward_spec,
                                          discounts=discount_spec)

        # Logger.
        dir_path = f'{params.exp_path}/logs/{self._name}'
        self._logger = make_default_logger(directory=dir_path, label=self._name)
        agent_logger = make_default_logger(directory=dir_path, label=f'{self._name}-learning')

        networks = _make_networks(actions_dim=params.num_phases,
                                  state_dim=params.states.rank,
                                  policy_layers=params.policy_layers,
                                  critic_layers=params.critic_layers)

        self.agent = acme_agent.DDPG(environment_spec=env_spec,
                                    policy_network=networks['policy'],
                                    critic_network=networks['critic'],
                                    observation_network=networks['observation'],
                                    discount=params.discount_factor,
                                    batch_size=params.batch_size,
                                    prefetch_size=params.prefetch_size,
                                    target_update_period=params.target_update_period,
                                    min_replay_size=params.min_replay_size,
                                    max_replay_size=params.max_replay_size,
                                    samples_per_insert=params.samples_per_insert,
                                    n_step=params.n_step,
                                    sigma_init=params.sigma_init,
                                    sigma_final=params.sigma_final,
                                    sigma_schedule_timesteps=params.sigma_schedule_timesteps,
                                    clipping=params.clipping,
                                    logger=agent_logger,
                                    checkpoint=False,
        )

        # Observations counter.
        self._obs_counter = 0

    def get_stop(self):
        return self._stop

    def set_stop(self, stop):
        self._stop = stop

    def act(self, s):
        s = double_to_single_precision(np.array(s))

        # Make first observation.
        if self._obs_counter == 0:
            t_1 = dm_env.restart(s)
            self.agent.observe_first(t_1)

        # Select action.
        if self._stop:
            action = self.agent.deterministic_action(s)
        else:
            action = self.agent.select_action(s)

        self._obs_counter += 1

        return tuple(action)

    def update(self, _, a, r, s1):

        if not self._stop:

            a = double_to_single_precision(np.array(a))
            r = double_to_single_precision(r)
            s1 = double_to_single_precision(np.array(s1))
            d = double_to_single_precision(1.0)

            timestep = dm_env.transition(reward=r,
                                         observation=s1,
                                         discount=d)

            self.agent.observe(a, timestep)
            self.agent.update()

        # Log values.
        values = {
            'step': self._obs_counter,
            'reward': r,
        }
        for i in range(self._params.num_phases):
            values[f"action_p{i}"] = a[i]
        self._logger.write(values)

    def terminate(self):
        self.agent.tear_down()

    def save_checkpoint(self, path):
        checkpoint_file = "{0}/checkpoints/{1}/{2}.chkpt".format(
            path, self._obs_counter, self._name)

        print(f'Saved chkpt: {checkpoint_file}')

        self.agent.save(checkpoint_file)

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        chkpt_path = '{0}/{1}/{2}.chkpt'.format(chkpts_dir_path,
                                                    chkpt_num,
                                                    self._name)

        print(f'Loaded chkpt: {chkpt_path}')

        self.agent.load(chkpt_path)