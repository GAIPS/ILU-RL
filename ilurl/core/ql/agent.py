import os
import pickle
import numpy as np
from threading import Thread

from ilurl.utils.meta import MetaAgentQ
from ilurl.core.params import QLParams
from ilurl.core.ql.choice import choice_eps_greedy, choice_ucb
from ilurl.core.ql.define import dpq_tls
from ilurl.core.ql.update import dpq_update

from ilurl.core.ql.replay_buffer import ReplayBuffer

class QL(object): #, metaclass=MetaAgentQ): # TODO: make agents metaclass
    """
        Q-learning agent.
    """

    def __init__(self, ql_params, name):
        """Instantiate Q-Learning agent.

        PARAMETERS
        ----------

        * ql_params: ilurl.core.params.QLParams object
            Q-learning agent parameters

        * name: str

        REFERENCES

            [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018
            
        """
        self.name = name

        self.stop = False

        # Learning rate.
        self.alpha = ql_params.alpha

        # Action choice.
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

        # Boolean list that stores whether actions
        # were randomly picked (exploration) or not.
        self.explored = []

        # Boolean list that stores the newly visited states.
        self.visited_states = []

        # Float list that stores the distance between
        # Q-tables between updates.
        self.Q_distances = []

        # Epsilon-greedy (exploration rate).
        if self.choice_type in ('eps-greedy',):
            self.epsilon = ql_params.epsilon

        # UCB (extra-stuff).
        if self.choice_type in ('ucb',):
            self.c = ql_params.c
            self.decision_counter = 0
            self.actions_counter = {
                state: {
                    action: 1.0
                    for action in actions
                }
                for state, actions in self.Q.items()
            }

        # Replay buffer.
        self.replay_buffer = ql_params.replay_buffer
        if self.replay_buffer:
            self.batch_size = ql_params.replay_buffer_batch_size
            self.warm_up = ql_params.replay_buffer_warm_up

            self.memory = ReplayBuffer(ql_params.replay_buffer_size)

        # Updates counter.
        self.updates_counter = 0

    def act(self, s):
        if self.stop:
            # Argmax greedy choice.
            actions, values = zip(*self.Q[s].items())
            choosen, exp = choice_eps_greedy(actions, values, 0)
            self.explored.append(exp)
        else:
            
            if self.choice_type in ('eps-greedy',):
                actions, values = zip(*self.Q[s].items())

                num_state_visits = sum(self.state_action_counter[s].values())
                eps = 1 / (1 + num_state_visits)

                choosen, exp = choice_eps_greedy(actions, values, eps)
                self.explored.append(exp)

            elif self.choice_type in ('optimistic',):
                raise NotImplementedError

            elif self.choice_type in ('ucb',):
                self.decision_counter += 1 if not self.stop else 0
                choosen = choice_ucb(self.Q[s].items(),
                                     self.c,
                                     self.decision_counter,
                                     self.actions_counter[s])
                self.actions_counter[s][choosen] += 1 if not self.stop else 0
            else:
                raise NotImplementedError

        return choosen

    def update(self, s, a, r, s1):

        # Track the visited states.
        if sum(self.state_action_counter[s].values()) == 0:
            self.visited_states.append(s)
        else:
            self.visited_states.append(None)

        if not self.stop:

            if self.replay_buffer:
                self.memory.add(s,a,r,s1,0.0)

            # Update (state, action) counter.
            self.state_action_counter[s][a] += 1

            # Calculate learning rate.
            lr = 1 / np.power(1 + self.state_action_counter[s][a], 2/3)

            Q_old = self.Q[s][a]

            # Q-learning update.
            try:
                r = sum(r)
            except TypeError:
                pass
            dpq_update(self.gamma, lr, self.Q, s, a, r, s1)

            # Calculate Q-tables distance.
            dist = np.abs(Q_old - self.Q[s][a])
            self.Q_distances.append(dist)

            if self.replay_buffer and self.updates_counter > self.warm_up:

                samples = self.memory.sample(self.batch_size)

                for sample in range(self.batch_size):
                    s = tuple(samples[0][sample])
                    a = (samples[1][sample][0],)
                    r = samples[2][sample]
                    s1 = tuple(samples[3][sample])

                    # Q-learning update.
                    try:
                        r = sum(r)
                    except TypeError:
                        pass
                    dpq_update(self.gamma, lr, self.Q, s, a, r, s1)

            self.updates_counter += 1

    def save_checkpoint(self, path):
        """
        Save model's weights.

        PARAMETERS
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

    """ @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q """

    @property
    def stop(self):
        """Stops exploring"""
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop