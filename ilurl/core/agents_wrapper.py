from copy import deepcopy

import numpy as np

from ilurl.core.params import Bounds
from ilurl.loaders.parser import config_parser

from ilurl.core.ql.agent import QL
from ilurl.core.dqn.agent import DQN

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

        num_variables = len(mdp_params.states_labels)

        # TODO: Afterwards this needs to come from a config file 
        # telling what each agent is controlling.
        # TODO: ATM it assumes each agent controls 1 intersection.
        for tid in mdp_params.phases_per_traffic_light.keys():

            agent_params_ = deepcopy(agent_params)

            # Action space.
            actions_depth = mdp_params.num_actions[tid]
            agent_params_.actions = Bounds(1, actions_depth) # TODO

            # State space.
            num_phases = mdp_params.phases_per_traffic_light[tid]
            states_rank = num_phases * num_variables
            states_depth = len(mdp_params.category_counts) + 1
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

        self.agents = agents

    @property
    def stop(self):
        stops = [agent.stop for agent in self.agents]
        return all(stops)

    @stop.setter
    def stop(self, stop):
        for agent in self.agents:
            agent.stop = stop

    def act(self, state):

        choices = {}

        for tid, agent in self.agents.items():
            choices[tid] = int(agent.act(state[tid]))

        return choices

    def update(self, s, a, r, s1):
        for tid, agent in self.agents.items():
            s_, a_, r_, s1_ = s[tid], a[tid], r[tid], s1[tid]
            agent.update(s_, a_, r_, s1_)

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