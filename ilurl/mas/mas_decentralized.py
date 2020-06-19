from copy import deepcopy

from ilurl.core.params import Bounds
from ilurl.loaders.parser import config_parser
from ilurl.agents.agent_client import AgentClient
from ilurl.agents.agent_factory import AgentFactory
from ilurl.mas.mas_interface import MASInterface


class DecentralizedMAS(MASInterface):
    """
        Decentralized multi-agent system wrapper.
    """

    def __init__(self, 
                mdp_params,
                exp_path,
                seed):

        # Load agent parameters from config file (train.config).
        agent_type, agent_params = config_parser.parse_agent_params()

        # Create agents.
        agents = {}

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

            # Experience path.
            agent_params_.exp_path = exp_path

            # Name.
            agent_params_.name = tid

            # Seed.
            agent_params_.seed = seed

            # Create agent client.
            agents[tid] = AgentClient(AgentFactory.get(agent_type),
                                     params=agent_params_)

        self.agents = agents

    @property
    def stop(self):
        # Send requests.
        for (tid, agent) in self.agents.items():
            agent.get_stop()

        # Retrieve requests.
        stops = [agent.stop for (_, agent) in self.agents.items()]

        return all(stops)

    @stop.setter
    def stop(self, stop):
        # Send requests.
        for (tid, agent) in self.agents.items():
            agent.set_stop(stop)

        # Synchronize.
        for (tid, agent) in self.agents.items():
            agent.receive()

    def act(self, state):
        # Send requests.
        for (tid, agent) in self.agents.items():
            agent.act(state[tid])

        # Retrieve requests.
        choices = {tid: agent.receive()
                    for (tid, agent) in self.agents.items()}

        return choices

    def update(self, s, a, r, s1):
        # Send requests.
        for (tid, agent) in self.agents.items():
            agent.update(s[tid], a[tid], r[tid], s1[tid])

        # Synchronize.
        for (tid, agent) in self.agents.items():
            agent.receive()

    def save_checkpoint(self, path):
        # Send requests.
        for (tid, agent) in self.agents.items():
            agent.save_checkpoint(path)

        # Synchronize.
        for (tid, agent) in self.agents.items():
            agent.receive()

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        # Send requests.
        for (tid, agent) in self.agents.items():
            agent.load_checkpoint(chkpts_dir_path, chkpt_num)

        # Synchronize.
        for (tid, agent) in self.agents.items():
            agent.receive()

    def terminate(self):
        # Send terminate request for each agent.
        # (also waits for termination)
        for (tid, agent) in self.agents.items():
            agent.terminate()
