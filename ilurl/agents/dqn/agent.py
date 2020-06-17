import os
import numpy as np

import dm_env

import acme
from acme import specs
from acme.tf import networks

import tensorflow as tf

from ilurl.utils.default_logger import make_default_logger

from ilurl.agents.dqn import acme_agent
from ilurl.agents.agent_worker import AgentWorker
from ilurl.agents.agent_interface import AgentInterface

_TF_USE_GPU = False
_TF_NUM_THREADS = 32


class DQN(AgentWorker,AgentInterface):
    """
        DQN agent.
    """
    def __init__(self, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)

    def init(self, params, exp_path, name):

        if not _TF_USE_GPU:
            tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(_TF_NUM_THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(_TF_NUM_THREADS)

        self._name = name

        # Whether learning stopped.
        self.stop = False

        observation_spec = specs.Array(shape=(params.states.rank,),
                                  dtype=np.float32,
                                  name='obs'
        )
        action_spec = specs.DiscreteArray(dtype=int,
                                          num_values=params.actions.depth,
                                          name="action"
        )
        reward_spec = specs.Array(shape=(),
                                  dtype=float,
                                  name='reward'
        ) # Default.
        discount_spec = specs.BoundedArray(shape=(),
                                           dtype=float,
                                           minimum=0.,
                                           maximum=1.,
                                           name='discount'
        ) # Default.
 
        env_spec = specs.EnvironmentSpec(observations=observation_spec,
                                          actions=action_spec,
                                          rewards=reward_spec,
                                          discounts=discount_spec)

        # Logger.
        dir_path = f'{exp_path}/logs/{self._name}'
        self._logger = make_default_logger(directory=dir_path, label=self._name)

        agent_logger = make_default_logger(directory=dir_path, label=f'{self._name}-learning')
        network = networks.duelling.DuellingMLP(num_actions=env_spec.actions.num_values,
                                                hidden_sizes=[8]) # TODO: FIX NETWORK PARAMS
        self.agent = acme_agent.DQN(environment_spec=env_spec,
                                    network=network,
                                    batch_size=params.batch_size,
                                    prefetch_size=params.prefetch_size,
                                    target_update_period=params.target_update_period,
                                    samples_per_insert=params.samples_per_insert,
                                    min_replay_size=params.min_replay_size,
                                    max_replay_size=params.max_replay_size,
                                    importance_sampling_exponent=params.importance_sampling_exponent,
                                    priority_exponent=params.priority_exponent,
                                    n_step=params.n_step,
                                    epsilon=params.epsilon,
                                    learning_rate=params.learning_rate,
                                    discount=params.gamma,
                                    logger=agent_logger)

        # Observations counter.
        self._obs_counter = 0

    def get_stop(self):
        return self._stop

    def set_stop(self, stop):
        self._stop = stop

    def act(self, s):
        s = np.array(s, dtype=np.float32)

        # Make first observation.
        if self._obs_counter == 0:
            t_1 = dm_env.restart(s)
            self.agent.observe_first(t_1)

        # Select action.
        if self.stop:
            action = self.agent.deterministic_action(s)
        else:
            action = self.agent.select_action(s)

        self._obs_counter += 1

        return int(action)

    def update(self, _, a, r, s1):
        if not self.stop:

            s1 = np.array(s1, dtype=np.float32)
            timestep = dm_env.transition(reward=r, observation=s1)

            self.agent.observe(a, timestep)
            self.agent.update()

            # Log values.
            values = {
                'step': self._obs_counter,
                'action': a,
                'reward': r,
            }
            self._logger.write(values)

    def save_checkpoint(self, path):
        checkpoint_file = "{0}/checkpoints/{1}/{2}.chkpt".format(
            path, self._obs_counter, self._name)

        print('SAVED')
        print(checkpoint_file)

        self.agent.save(checkpoint_file)

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        chkpt_path = '{0}/{1}/{2}.chkpt'.format(chkpts_dir_path,
                                                    chkpt_num,
                                                    self._name)

        print('LOADED')
        print(chkpt_path)

        self.agent.load(chkpt_path)
