import os
import numpy as np

from ilurl.utils.meta import MetaAgent

import tensorflow as tf

# Suppress tf warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import tensorflow.contrib.layers as layers

from gym.spaces.box import Box

import baselines.common.tf_util as U

from baselines.logger import Logger, TensorBoardOutputFormat, CSVOutputFormat
from baselines import deepq
from baselines.common.tf_util import load_variables, save_variables
from baselines.deepq.deepq import ActWrapper
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

# TODO: allow for arbitrary network architecture.
def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class DQN(object, metaclass=MetaAgent):
    """
        DQN agent.
    """

    def __init__(self, params, name):
        """Instantiate DQN agent.

        PARAMETERS
        ----------
        * params: ilurl.core.params.DQNParams object.
            object containing DQN parameters.

        * name: str

        """
        self.name = name

        # Whether learning stopped.
        self.stop = False

        # Parameters.
        self.model = model # TODO: allow for arbitrary network architecture.
        self.lr = params.lr
        self.gamma = params.gamma
        self.buffer_size = params.buffer_size
        self.batch_size = params.batch_size
        self.exp_initial_p = params.exp_initial_p
        self.exp_final_p = params.exp_final_p
        self.exp_schedule_timesteps = params.exp_schedule_timesteps
        self.learning_starts = params.learning_starts
        self.target_net_update_interval = params.target_net_update_interval

        # Observation space.
        self.obs_space = Box(low=np.array([0.0]*params.states.rank),
                             high=np.array([np.inf]*params.states.rank),
                             dtype=np.float64)
        def make_obs_ph(name):
            return ObservationInput(self.obs_space, name=name)

        # Action space.
        self.num_actions = params.actions.depth # TODO

        self.action, self.train, self.update_target, self.debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=self.model,
            num_actions=self.num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.lr),
            gamma=self.gamma,
            scope=self.name
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

        # Tensorboard logger.
        self.logger = None

        # Updates counter.
        self.updates_counter = 0

    @property
    def stop(self):
        """Stops learning."""
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop

    def act(self, s):
        """
        Specify the actions to be performed by the RL agent(s).

        Parameters:
        ----------
        * s: tuple
            state representation.

        """
        s = np.array(s)

        if self.stop:
            action, _ = self.action(s[None], stochastic=False)
        else:
            action, _ = self.action(s[None],
                              update_eps=self.exploration.value(self.updates_counter))

        return action[0]

    def update(self, s, a, r, s1):
        """
        Performs a learning update step.

        Parameters:
        ----------
        * s: tuple 
            state representation.

        * a: int
            action.

        * r: float
            reward.

        * s1: tuple
            state representation.

        """
        if not self.stop:

            s = np.array(s)
            s1 = np.array(s1)

            self.replay_buffer.add(s, a, r, s1, 0.0)

            if self.updates_counter > self.learning_starts:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                td_errors = self.train(obses_t,
                        actions,
                        rewards,
                        obses_tp1,
                        dones,
                        np.ones_like(rewards))

                if self.logger:
                    self.logger.logkv("td_error", np.mean(td_errors))

            # Update target network.
            if self.updates_counter % self.target_net_update_interval == 0:
                self.update_target()

            # Tensorboard log.
            if self.logger:
                self.logger.logkv("action", a)
                self.logger.logkv("reward", r)
                self.logger.logkv("step", self.updates_counter)
                self.logger.logkv("lr", self.lr)
                self.logger.logkv("expl_eps",
                                    self.exploration.value(self.updates_counter))                    

                self.logger.dumpkvs()

            self.updates_counter += 1

    def save_checkpoint(self, path):
        """
        Save model's weights.

        Parameters:
        ----------
        * path: str 
            path to save directory.

        """
        if (self.updates_counter > self.learning_starts):
            checkpoint_file = '{0}/checkpoints/{1}-{2}.chkpt'.format(path,
                                                        self.name,
                                                        self.updates_counter)
            save_variables(checkpoint_file, scope_name=self.name)

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        """
        Loads model's weights from file.
 
        Parameters:
        ----------
        * chkpts_dir_path: str
            path to checkpoint's directory.

        * chkpt_num: int
            the number of the checkpoint to load.

        """
        chkpt_path = '{0}/{1}-{2}.chkpt'.format(chkpts_dir_path,
                                                    self.name,
                                                    chkpt_num)

        load_variables(chkpt_path, scope_name=self.name)

    def setup_logger(self, path):
        """
        Setup train logger (tensorboard).
 
        Parameters:
        ----------
        * path: str 
            path to log directory.

        """
        os.makedirs(f"{path}/train_logs", exist_ok=True)

        log_file = f'{path}/train_logs/{self.name}'
        tb_logger = TensorBoardOutputFormat(log_file)
        csv_logger = CSVOutputFormat(f'{log_file}.csv')
        self.logger = Logger(dir=path, output_formats=[tb_logger, csv_logger])
