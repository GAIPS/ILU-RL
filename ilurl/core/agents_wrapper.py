from copy import deepcopy
import multiprocessing

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
                mdp_params,
                exp_path):

        # Load agent parameters from config file (train.config).
        agent_type, agent_params = config_parser.parse_agent_params()

        # Create agents.
        #multiprocessing.set_start_method('spawn')
        mp = multiprocessing.get_context('spawn')

        pipes = {}
        agents = {}

        for tid in mdp_params.phases_per_traffic_light.keys():

            # Agents factory.
            if agent_type == 'QL':
                pass
                #agents[tid] = QL(agent_params_, exp_path, name=tid)
            elif agent_type == 'DQN':
                comm_pipe = mp.Pipe()
                agents[tid] = DQN(comm_pipe[1])
                pipes[tid] = comm_pipe[0]
            else:
                raise ValueError(f'''
                Agent type must be in {AGENT_TYPES}.
                Got {agent_type} type instead.''')

        for tid in mdp_params.phases_per_traffic_light.keys():
            agents[tid].start()

        print('AGENTS WRAPPER: AGENTS STARTED')
        print('Agents', agents)
        print('Pipes', pipes)

        num_variables = len(mdp_params.states_labels)
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
                pass
                #agents[tid] = QL(agent_params_, exp_path, name=tid)
            elif agent_type == 'DQN':
                args = (agent_params_, exp_path, tid)
                pipes[tid].send(('init', args))
            else:
                raise ValueError(f'''
                Agent type must be in {AGENT_TYPES}.
                Got {agent_type} type instead.''')

        # Synchronize.
        for tid in mdp_params.phases_per_traffic_light.keys():
            pipes[tid].recv()

        self.agents = agents
        self.pipes = pipes

    @property
    def stop(self):
        stops = [agent.stop for agent in self.agents]
        return all(stops)

    @stop.setter
    def stop(self, stop):
        for agent in self.agents.values():
            agent.stop = stop

    def act(self, state):

        choices = {}

        for tid, agent in self.agents.items():
            args = (state[tid],)
            self.pipes[tid].send(('act', args))

        for tid, agent in self.agents.items():
            choices[tid] = int(self.pipes[tid].recv())

        return choices

    def update(self, s, a, r, s1):
        for tid, agent in self.agents.items():
            #s_, a_, r_, s1_ = s[tid], a[tid], r[tid], s1[tid]
            #agent.update(s_, a_, r_, s1_)
            args = (s[tid], a[tid], r[tid], s1[tid])
            self.pipes[tid].send(('update', args))

        for tid, agent in self.agents.items():
            self.pipes[tid].recv()


    def save_checkpoint(self, path):
        """
        Save models' weights.

        Parameters:
        ----------
        * path: str 
            path to save directory.

        """
        for tid, agent in self.agents.items():
            args = (path)
            self.pipes[tid].send('save_checkpoint', args)
            #agent.save_checkpoint(path)

        for tid, agent in self.agents.items():
            self.pipes[tid].recv()

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
        for tid, agent in self.agents.items():
            args = (chkpts_dir_path, chkpt_num)
            self.pipes[tid].send('load_checkpoint', args)
            #agent.load_checkpoint(chkpts_dir_path, chkpt_num)

        for tid, agent in self.agents.items():
            self.pipes[tid].recv()

    # def setup_logs(self, path):
    #     """
    #     Setup train loggers (tensorboard).
 
    #     Parameters:
    #     ----------
    #     * path: str 
    #         path to log directory.

    #     """
    #     for agent in self.agents.values():
    #         agent.setup_logger(path)
