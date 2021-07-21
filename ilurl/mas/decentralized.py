from copy import deepcopy

from ilurl.params import Bounds
from ilurl.loaders.parser import config_parser
from ilurl.agents.client import AgentClient
from ilurl.agents.factory import AgentFactory
from ilurl.interfaces.mas import MASInterface


class DecentralizedMAS(MASInterface):
    """
        Decentralized multi-agent system wrapper.
    """

    def __init__(self, mdp_params, exp_path, seed):

        # Load agent parameters from config file (train.config).
        agent_type, agent_params = config_parser.parse_agent_params()

        # Create agents.
        agents = {}

        num_variables = len(mdp_params.features)

        # State space.
        # Period is a "Global" state space
        has_period = mdp_params.time_period is not None
        states_depth = None # Assign only when discretize_state_space
        for tid in mdp_params.phases_per_traffic_light:
            agent_params_ = deepcopy(agent_params)
            num_phases = mdp_params.phases_per_traffic_light[tid]
            # Action space.
            if mdp_params.action_space == 'discrete':
                # Discrete action space.
                # In the discrete action space each agent is allowed to pick
                # the signal plan(s) from a set of candidate signal plans given
                # a priori to the system by a user. The candidate signal plans
                # are defined in data/networks/{NETWORK_NAME}/tls_config.json
                actions_depth = mdp_params.num_actions[tid]
                agent_params_.actions = Bounds(rank=1,
                                               depth=actions_depth)
            else:
                # Continuous action space.
                # In the continuous action space each agent is allowed to select
                # the portion of the cycle length allocated for each of the phases.
                agent_params_.num_phases = mdp_params.phases_per_traffic_light[tid]

            if mdp_params.discretize_state_space:
                feature = mdp_params.features[0]
                states_depth = len(mdp_params.categories[tid][feature]['0']) + 1

            # states_rank = num_phases * num_variables + int(has_period)
            states_rank = num_phases * num_variables + 2
            agent_params_.states = Bounds(states_rank, states_depth)
            # Experience path.
            agent_params_.exp_path = exp_path

            # Name.
            agent_params_.name = tid

            # Seed.
            agent_params_.seed = seed

            # Discount factor (gamma).
            agent_params_.discount_factor = mdp_params.discount_factor

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
