import os
import random
from typing import Sequence

import numpy as np

import dm_env

import tensorflow as tf
import sonnet as snt

import acme
from acme import specs


from ilurl.agents.dqn import acme_agent
from ilurl.agents.worker import AgentWorker
from ilurl.interfaces.agents import AgentInterface
from ilurl.utils.default_logger import make_default_logger
from ilurl.utils.precision import double_to_single_precision

import ilurl.agents.dqn.duelling as duelling

_TF_USE_GPU = False
_TF_NUM_THREADS = 32


def _make_network(num_actions : int,
                  torso_layers : Sequence[int] = [5],
                  head_layers  : Sequence[int] = [5]):
    
    network = snt.Sequential([
        # Torso MLP.
        snt.nets.MLP(torso_layers, activate_final=True),
        # Dueling MLP head.
        duelling.DuellingMLP(num_actions=num_actions,
                                      hidden_sizes=head_layers)  
    ])
    
    
    return network


class DQN(AgentWorker,AgentInterface):
    """
        DQN agent.
    """
    def __init__(self, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)

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
        action_spec = specs.DiscreteArray(dtype=np.int32,
                                          num_values=params.actions.depth,
                                          name="action"
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

        self.network = _make_network(num_actions=env_spec.actions.num_values,
                                torso_layers=params.torso_layers,
                                head_layers=params.head_layers)


        self.agent = acme_agent.DQN(environment_spec=env_spec,
                                    network=self.network,
                                    batch_size=params.batch_size,
                                    prefetch_size=params.prefetch_size,
                                    target_update_period=params.target_update_period,
                                    samples_per_insert=params.samples_per_insert,
                                    min_replay_size=params.min_replay_size,
                                    max_replay_size=params.max_replay_size,
                                    importance_sampling_exponent=params.importance_sampling_exponent,
                                    priority_exponent=params.priority_exponent,
                                    n_step=params.n_step,
                                    epsilon_init=params.epsilon_init,
                                    epsilon_final=params.epsilon_final,
                                    epsilon_schedule_timesteps=params.epsilon_schedule_timesteps,
                                    learning_rate=params.learning_rate,
                                    discount=params.discount_factor,
                                    logger=agent_logger)



        # Observations counter.
        self._obs_counter = 0

    def get_stop(self):
        return self._stop

    def set_stop(self, stop):
        self._stop = stop

    def forward_pass(self, state):
        return self.network(tf.constant([state]))

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

        return int(action)

    def update(self, _, a, r, s1):

        if not self._stop:

            a = double_to_single_precision(a)
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
            'action': a,
            'reward': r,
        }
        self._logger.write(values)

    def terminate(self):
        self.agent.tear_down()

    def save_checkpoint(self, path):
        checkpoint_file = "{0}/checkpoints/{1}/{2}.chkpt".format(
            path, self._obs_counter, self._name)

        print(f'Saved chkpt: {checkpoint_file}')

        self.agent.save(checkpoint_file)

    def save_checkpoint(self, path, chkpt_num):
        checkpoint_file = "{0}/checkpoints/{1}/{2}.chkpt".format(
            path, chkpt_num, self._name)

        print(f'Saved chkpt: {checkpoint_file}')

        self.agent.save(checkpoint_file)

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        chkpt_path = '{0}/{1}/{2}.chkpt'.format(chkpts_dir_path,
                                                    chkpt_num,
                                                    self._name)

        print(f'Loaded chkpt: {chkpt_path}')

        self.agent.load(chkpt_path)
