import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from gym.spaces.box import Box

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.common.tf_util import load_variables, save_variables
from baselines.deepq.deepq import ActWrapper
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=8, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class DQN(object):

    def __init__(self,
                n_actions,
                ql_params,
                log_dir=None,
                checkpoints_dir=None,
                load_path=None
                ):
        """
            DQN agent.
        """

        # Whether learning stopped.
        self.stop = False

        self.ql_params = ql_params

        # Parameters.
        self.model = model
        self.lr = 5e-4
        self.gamma = 0.8
        self.buffer_size = 20000
        self.batch_size = 64
        self.exp_initial_p = 1.0
        self.exp_final_p = 0.02
        self.exp_schedule_timesteps = 40000
        self.learning_starts = 2000
        self.target_net_update_interval = 2000

        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_freq = 5000
        self.load_path = load_path
        self.log_dir = log_dir

        # Observation space.
        self.obs_space = Box(low=np.array([0.0, 0.0, 0.0, 0.0]),
                             high=np.array([20.0, 30.0, 20.0, 30.0]),
                             dtype=np.float64)
        def make_obs_ph(name):
            return ObservationInput(self.obs_space, name=name)

        # Action space.
        self.num_actions = n_actions

        self.action, self.train, self.update_target, self.debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=self.model,
            num_actions=self.num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.lr),
            gamma=self.gamma,
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': self.model,
            'num_actions': self.num_actions,
        }
        self.action = ActWrapper(self.action, act_params)

        # TODO: add option for prioritized replay buffer.
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.exploration = LinearSchedule(schedule_timesteps=self.exp_schedule_timesteps,
                                          initial_p=self.exp_initial_p,
                                          final_p=self.exp_final_p)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        self.update_target()

        # Load model if load_path is set.
        if self.load_path is not None:
            load_variables(self.load_path)

        # Setup tensorboard.
        if self.log_dir is not None:
            logger.configure(dir=self.log_dir,
                            format_strs=['csv', 'tensorboard'])

        # Boolean list that stores whether actions
        # were randomly picked (exploration) or not.
        self.explored = []

        self.updates_counter = 0

    @property
    def stop(self):
        """Stops exploring"""
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop

    def act(self, s):
        """
        Specify the actions to be performed by the RL agent(s).

        Parameters
        ----------
        s: tuple with state representation

        """

        s = np.array(list(s))

        if self.stop:
            action, explored = self.action(s[None],
                              stochastic=False)
        else:
            action, explored = self.action(s[None],
                              update_eps=self.exploration.value(self.updates_counter))

        self.explored.append(explored[0])

        return (action[0],)

    def update(self, s, a, r, s1):

        if not self.stop:

            s = np.array(list(s))
            s1 = np.array(list(s1))
            a = a[0]
            r = r[0]

            self.replay_buffer.add(s, a, r, s1, 0.0)

            if self.updates_counter > self.learning_starts:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                td_errors = self.train(obses_t,
                        actions,
                        rewards,
                        obses_tp1,
                        dones,
                        np.ones_like(rewards))
                
                td_error = np.mean(td_errors)
                logger.record_tabular("td_error", td_error)

            if self.updates_counter % self.target_net_update_interval == 0:
                self.update_target()

            logger.record_tabular("step", self.updates_counter)
            logger.record_tabular("expl_eps",
                                self.exploration.value(self.updates_counter))
            logger.dump_tabular()

            if (self.checkpoints_dir is not None and 
                self.updates_counter > self.learning_starts and
                self.updates_counter % self.checkpoints_freq == 0):
                checkpoint_file = '{0}checkpoint-{1}'.format(self.checkpoints_dir, 
                                                            self.updates_counter)
                logger.log('Saved model to {}'.format(checkpoint_file))
                save_variables(checkpoint_file)

            self.updates_counter += 1