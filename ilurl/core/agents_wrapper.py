from copy import deepcopy

import numpy as np

from ilurl.core.params import Bounds
from ilurl.loaders.parser import config_parser

from ilurl.core.ql.agent import QL
from ilurl.core.dqn.agent import DQN

import baselines.common.tf_util as U

AGENT_TYPES = ('QL', 'DQN')

class AgentsWrapper(object):
    """
        Multi-agent system wrapper.
    """

    def __init__(self,
                mdp_params):

        # Load agent parameters from config file (train.config).
        agent_type, agent_params = config_parser.parse_agent_params()

        # Create agents.
        agents = {}

        self.neighbours = mdp_params.neighbours

        num_variables = len(mdp_params.states_labels)

        U.reset_session()

        for tid in mdp_params.phases_per_traffic_light.keys():

            agent_params_ = deepcopy(agent_params)

            # Action space.
            actions_depth = mdp_params.num_actions[tid]

            agent_params_.actions = Bounds(1, actions_depth) # TODO

            # State space.
            num_phases = mdp_params.phases_per_traffic_light[tid]
            states_rank = num_phases * num_variables
            states_depth = len(mdp_params.category_counts) + 1

            # Calculate new network input size.
            states_rank = states_rank * 2 + actions_depth            

            agent_params_.states = Bounds(states_rank, states_depth)

            # Agents factory.
            if agent_type == 'QL':
                agents[tid] = QL(agent_params_, name=tid)
            elif agent_type == 'DQN':
                agents[tid] = DQN(agent_params_, name=tid)
            else:
                raise ValueError(f'''
                Agent type must be in {AGENT_TYPES}.
                Got {agent_type} type instead.''')

        # Agents.
        self.agents = agents

        # Previous action.
        self.prev_action = {tid: 0 for tid in self.agents}

        # Number of actions (homegeneous society).
        self.num_actions = actions_depth

    @property
    def stop(self):
        stops = [agent.stop for agent in self.agents]
        return all(stops)

    @stop.setter
    def stop(self, stop):
        for agent in self.agents.values():
            agent.stop = stop

    def act(self, state):

        # Output action.
        choices = {}

        # Calculate average neighbours state for each agent.
        neighbours_avg_states = {}
        for tid, agent in self.agents.items():

            # Calculate average neighbours state.
            neighbours_states = [state[n] for n in self.neighbours[tid]]
            neighbours_states = np.array(neighbours_states)
            neighbours_avg_states[tid] = np.mean(neighbours_states, axis=0)

        M = 10
        for _ in range(M):

            for tid, agent in self.agents.items():

                # Calculate average neighbours actions.
                neighbours_actions = [self.prev_action[n] for n in self.neighbours[tid]]
                neighbours_actions = np.array(neighbours_actions)
                one_hot_actions = np.zeros((neighbours_actions.size, self.num_actions))
                one_hot_actions[np.arange(neighbours_actions.size),neighbours_actions] = 1
                neighbours_avg_action = np.mean(one_hot_actions, axis=0)

                # Prepare network input.
                state_contat = tuple(np.hstack((state[tid],
                                                neighbours_avg_states[tid],
                                                neighbours_avg_action)))

                # Pick agent's action.
                choices[tid] = int(agent.act(state_contat))

            self.prev_action = choices

        return choices

    def update(self, s, a, r, s1):

        for tid, agent in self.agents.items():

            # Concatenate neighbours' information.
            neighbours_states = [s[n] for n in self.neighbours[tid]]
            neighbours_actions = [a[n] for n in self.neighbours[tid]]
            # neighbours_rewards = [r[n] for n in self.neighbours[tid]]
            neighbours_next_states = [s1[n] for n in self.neighbours[tid]]

            # Calculate average neighbours state.
            neighbours_states = np.array(neighbours_states)
            neighbours_avg_state = np.mean(neighbours_states, axis=0)

            # Calculate average neighbours actions.
            neighbours_actions = np.array(neighbours_actions)
            one_hot_actions = np.zeros((neighbours_actions.size, self.num_actions))
            one_hot_actions[np.arange(neighbours_actions.size),neighbours_actions] = 1
            neighbours_avg_action = np.mean(one_hot_actions, axis=0)

            # Calculate average neighbours next state.
            neighbours_next_states = np.array(neighbours_next_states)
            neighbours_avg_next_state = np.mean(neighbours_next_states, axis=0)

            # Prepare network input.
            # (agent's state + neighbours_avg_state + neighbours_avg_action).
            state = tuple(np.hstack((s[tid],
                                     neighbours_avg_state,
                                     neighbours_avg_action)))

            # Calculate agent's reward.
            # reward = r[tid] + np.mean(neighbours_rewards)
            reward = r[tid] + np.sum(list(r.values()))


            # Prepare target network input (next state).
            # (agent's state + neighbours_avg_state + neighbours_avg_action).
            next_state = tuple(np.hstack((s1[tid],
                               neighbours_avg_next_state,
                               neighbours_avg_action)))

            # Update agent.
            agent.update(state, a[tid], reward, next_state)

    def save_checkpoint(self, path):
        """
        Save models' weights.

        Parameters:
        ----------
        * path: str 
            path to save directory.

        """
        for agent in self.agents.values():
            agent.save_checkpoint(path)

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        """
        Loads models' weights from files.
 
        Parameters:
        ----------
        * chkpts_dir_path: str
            path to checkpoints' directory.

        * chkpt_num: int
            the number of the checkpoints to load.

        """
        for agent in self.agents.values():
            agent.load_checkpoint(chkpts_dir_path, chkpt_num)

    def setup_logs(self, path):
        """
        Setup train loggers (tensorboard).
 
        Parameters:
        ----------
        * path: str 
            path to log directory.

        """
        for agent in self.agents.values():
            agent.setup_logger(path)
