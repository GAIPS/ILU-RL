import os
import pickle
import numpy as np
from threading import Thread

from ilurl.utils.meta import MetaAgent
from ilurl.core.ql.choice import choice_eps_greedy, choice_ucb
from ilurl.core.ql.define import dpq_tls
from ilurl.core.ql.update import dpq_update

from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.logger import Logger, TensorBoardOutputFormat
from baselines.common.schedules import PowerSchedule

class QL(object, metaclass=MetaAgent):
    """
        Q-learning agent.
    """

    def __init__(self, ql_params, name):
        """Instantiate Q-Learning agent.

        Parameters:
        ----------

        * ql_params: ilurl.core.params.QLParams object
            Q-learning agent parameters

        * name: str

        References:
        ----------

        [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018
            
        """
        self.name = name

        # Whether learning stopped.
        self.stop = False

        # Learning rate.
        self.learning_rate = PowerSchedule(
                        power_coef=ql_params.lr_decay_power_coef)

        # Exploration strategy.
        self.choice_type = ql_params.choice_type

        # Discount factor.
        self.gamma = ql_params.gamma

        # Q-table.
        self.Q = dpq_tls(ql_params.states.rank, ql_params.states.depth,
                         ql_params.actions.rank, ql_params.actions.depth,
                         ql_params.initial_value)

        # State-action counter (for learning rate decay).
        self.state_action_counter = dpq_tls(ql_params.states.rank,
                                            ql_params.states.depth,
                                            ql_params.actions.rank,
                                            ql_params.actions.depth,
                                            0)

        # Epsilon-greedy (exploration rate).
        if self.choice_type in ('eps-greedy',):
            self.exploration = PowerSchedule(
                        power_coef=ql_params.eps_decay_power_coef)

        # UCB (extra-stuff).
        if self.choice_type in ('ucb',):
            raise NotImplementedError
            # self.c = ql_params.c
            # self.decision_counter = 0
            # self.actions_counter = {
            #     state: {
            #         action: 1.0
            #         for action in actions
            #     }
            #     for state, actions in self.Q.items()
            # }

        # Replay buffer.
        self.replay_buffer = ql_params.replay_buffer
        if self.replay_buffer:
            self.batch_size = ql_params.replay_buffer_batch_size
            self.warm_up = ql_params.replay_buffer_warm_up

            self.memory = ReplayBuffer(ql_params.replay_buffer_size)

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
        if self.stop:
            # Argmax greedy choice.
            actions, values = zip(*self.Q[s].items())
            choosen, _ = choice_eps_greedy(actions, values, 0)
        else:
            
            if self.choice_type in ('eps-greedy',):
                actions, values = zip(*self.Q[s].items())

                num_state_visits = sum(self.state_action_counter[s].values())

                eps = self.exploration.value(num_state_visits)
                choosen, _ = choice_eps_greedy(actions, values, eps)

            elif self.choice_type in ('optimistic',):
                raise NotImplementedError

            elif self.choice_type in ('ucb',):
                raise NotImplementedError
                # self.decision_counter += 1 if not self.stop else 0
                # choosen = choice_ucb(self.Q[s].items(),
                #                      self.c,
                #                      self.decision_counter,
                #                      self.actions_counter[s])
                # self.actions_counter[s][choosen] += 1 if not self.stop else 0
            else:
                raise NotImplementedError

        return choosen

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

            if self.replay_buffer:
                self.memory.add(s,a,r,s1,0.0)

            # Update (state, action) counter.
            self.state_action_counter[s][a] += 1

            # Calculate learning rate.
            lr = self.learning_rate.value(self.state_action_counter[s][a])

            Q_old = self.Q[s][a]

            # Q-learning update.
            dpq_update(self.gamma, lr, self.Q, s, a, r, s1)

            # Calculate Q-table update distance.
            dist = np.abs(Q_old - self.Q[s][a])

            if self.replay_buffer and self.updates_counter > self.warm_up:

                s_samples, a_samples, r_samples, s1_samples, _ = self.memory.sample(
                                                                self.batch_size)

                for sample in range(self.batch_size):
                    s_ = tuple(s_samples[sample])
                    a_ = a_samples[sample]
                    r_ = r_samples[sample]
                    s1_ = tuple(s1_samples[sample])

                    # Q-learning update.
                    dpq_update(self.gamma, lr, self.Q, s_, a_, r_, s1_)

            # Tensorboard log.
            if self.logger:
                self.logger.logkv("action", a)
                self.logger.logkv("reward", r)
                self.logger.logkv("step", self.updates_counter)
                self.logger.logkv("lr", lr)
                self.logger.logkv("expl_eps", self.exploration.value(
                    sum(self.state_action_counter[s].values())-1))

                self.logger.logkv("q_dist", dist)

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
        os.makedirs(f"{path}/checkpoints", exist_ok=True)

        checkpoint_file = "{0}/checkpoints/{1}-{2}.pickle".format(
            path, self.name, self.updates_counter)

        with open(checkpoint_file, 'wb') as f:
            t = Thread(target=pickle.dump(self.Q, f))
            t.start()

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
        self.logger = Logger(dir=path, output_formats=[tb_logger])