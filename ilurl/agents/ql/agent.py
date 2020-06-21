import os
import dill
import pickle
import numpy as np
from threading import Thread

from ilurl.utils.default_logger import make_default_logger

from ilurl.agents.worker import AgentWorker
from ilurl.interfaces.agents import AgentInterface

from ilurl.agents.ql.choice import choice_eps_greedy, choice_ucb
from ilurl.agents.ql.define import dpq_tls
from ilurl.agents.ql.update import dpq_update
from ilurl.agents.ql.replay_buffer import ReplayBuffer
from ilurl.agents.ql.schedules import PowerSchedule


class QL(AgentWorker,AgentInterface):
    """
        Q-learning agent.
    """
    def __init__(self, *args, **kwargs):
        super(QL, self).__init__(*args, **kwargs)

    def init(self, ql_params):
        """Instantiate Q-Learning agent.

        Parameters:
        ----------

        * ql_params: ilurl.core.params.QLParams object
            Q-learning agent parameters

        References:
        ----------

        [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018
            
        """
        self._name = ql_params.name

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

        # Replay buffer.
        self.replay_buffer = ql_params.replay_buffer
        if self.replay_buffer:
            self.batch_size = ql_params.replay_buffer_batch_size
            self.warm_up = ql_params.replay_buffer_warm_up

            self.memory = ReplayBuffer(ql_params.replay_buffer_size)

        # Logger.
        dir_path = f'{ql_params.exp_path}/logs/{self._name}'
        self._logger = make_default_logger(directory=dir_path, label=self._name)

        # Observations counter.
        self._obs_counter = 0

    def get_stop(self):
        return self._stop

    def set_stop(self, stop):
        self._stop = stop

    def act(self, s):
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
            else:
                raise NotImplementedError

        self._obs_counter += 1

        return int(choosen)

    def update(self, s, a, r, s1):
        if not self.stop:

            if self.replay_buffer:
                self.memory.add(s, a, r, s1, 0.0)

            # Update (state, action) counter.
            self.state_action_counter[s][a] += 1

            # Calculate learning rate.
            lr = self.learning_rate.value(self.state_action_counter[s][a])

            Q_old = self.Q[s][a]

            # Q-learning update.
            dpq_update(self.gamma, lr, self.Q, s, a, r, s1)

            # Calculate Q-table update distance.
            dist = np.abs(Q_old - self.Q[s][a])

            if self.replay_buffer and self._obs_counter > self.warm_up:

                s_samples, a_samples, r_samples, s1_samples, _ = self.memory.sample(
                                                                self.batch_size)

                for sample in range(self.batch_size):
                    s_ = tuple(s_samples[sample])
                    a_ = a_samples[sample]
                    r_ = r_samples[sample]
                    s1_ = tuple(s1_samples[sample])

                    # Q-learning update.
                    dpq_update(self.gamma, lr, self.Q, s_, a_, r_, s1_)

            # Log values.
            values = {
                "action": a,
                "reward": r,
                "step": self._obs_counter,
                "lr": lr,
                "expl_eps": self.exploration.value(sum(
                        self.state_action_counter[s].values())-1),
                "q_dist": dist,
            }
            self._logger.write(values)

    def save_checkpoint(self, path):
        os.makedirs(f"{path}/checkpoints/{self._obs_counter}", exist_ok=True)

        checkpoint_file = "{0}/checkpoints/{1}/{2}.chkpt".format(
            path, self._obs_counter, self._name)

        print('SAVED')
        print(checkpoint_file)

        with open(checkpoint_file, 'wb') as f:
            t = Thread(target=pickle.dump(self.Q, f))
            t.start()

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        chkpt_path = '{0}/{1}/{2}.chkpt'.format(chkpts_dir_path,
                                                    chkpt_num,
                                                    self._name)

        print('LOADED')
        print(chkpt_path)

        with open(chkpt_path, 'rb') as f:
            self.Q =  dill.load(f)

