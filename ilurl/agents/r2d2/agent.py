import os
import random
import numpy as np

import tensorflow as tf
import sonnet as snt

import dm_env
import acme
from acme import specs
from acme.tf import networks

from ilurl.agents.r2d2 import acme_agent
from ilurl.agents.worker import AgentWorker
from ilurl.interfaces.agents import AgentInterface
from ilurl.utils.default_logger import make_default_logger
from ilurl.utils.precision import double_to_single_precision

_TF_USE_GPU = False
_TF_NUM_THREADS = 1


# TODO: Allow for dynamic network creation via user parameters.
class SimpleNetwork(networks.RNNCore):

  def __init__(self, action_spec: specs.DiscreteArray):
    super().__init__(name='r2d2_network')
    self._net = snt.DeepRNN([
        snt.Flatten(),
        snt.LSTM(10),
        snt.nets.MLP([5, action_spec.num_values])
    ])

  def __call__(self, inputs, state):
    return self._net(inputs, state)

  def initial_state(self, batch_size: int, **kwargs):
    return self._net.initial_state(batch_size)

  def unroll(self, inputs, state, sequence_length):
    return snt.static_unroll(self._net, inputs, state, sequence_length)


class R2D2(AgentWorker,AgentInterface):
    """
        R2D2 agent.
    """
    def __init__(self, *args, **kwargs):
        super(R2D2, self).__init__(*args, **kwargs)

    def init(self, params):

        if not _TF_USE_GPU:
            tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(_TF_NUM_THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(_TF_NUM_THREADS)

        if params.seed:
            random.seed(params.seed)
            np.random.seed(params.seed)
            tf.random.set_seed(params.seed)

        # Internalize params.
        self._params = params

        self._name = params.name

        # Whether learning stopped.
        self.stop = False

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

        self.agent = acme_agent.R2D2(environment_spec=env_spec,
                                     network=SimpleNetwork(env_spec.actions), # TODO.
                                     batch_size=params.batch_size,
                                     samples_per_insert=params.samples_per_insert,
                                     burn_in_length=params.burn_in_length,
                                     trace_length=params.trace_length,
                                     replay_period=params.replay_period,
                                     min_replay_size=params.min_replay_size,
                                     max_replay_size=params.max_replay_size,
                                     discount=params.discount_factor,
                                     prefetch_size=params.prefetch_size,
                                     target_update_period=params.target_update_period,
                                     importance_sampling_exponent=params.importance_sampling_exponent,
                                     priority_exponent=params.priority_exponent,
                                     epsilon=params.epsilon,
                                     learning_rate=params.learning_rate,
                                     store_lstm_state=params.store_lstm_state,
                                     max_priority_weight=params.max_priority_weight,
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
        if self.stop:
            action = self.agent.deterministic_action(s)
        else:
            action = self.agent.select_action(s)

        self._obs_counter += 1

        return int(action)

    def update(self, _, a, r, s1):

        if not self.stop:

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
        # Fake a final transition.
        a = double_to_single_precision(0)
        s = double_to_single_precision(np.zeros((self._params.states.rank,)))
        r = double_to_single_precision(0.0)
        d = double_to_single_precision(0.0)

        end = dm_env.TimeStep(dm_env.StepType.LAST, r, d, s)

        self.agent.observe(a, end)

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
