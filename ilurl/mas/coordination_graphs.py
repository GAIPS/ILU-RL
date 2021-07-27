from copy import deepcopy

from ilurl.params import Bounds
from ilurl.loaders.parser import config_parser
from ilurl.agents.client import AgentClient
from ilurl.agents.factory import AgentFactory
from ilurl.interfaces.mas import MASInterface


class CoordinationGraphsMAS(MASInterface):
    """
        Multi-agent system with coordination graphs wrapper.
    """

    def __init__(self, mdp_params, exp_path, seed, network):

        # Load agent parameters from config file (train.config).
        agent_type, agent_params = config_parser.parse_agent_params()

        # Create agents.
        agents = {}
        num_variables = len(mdp_params.features)
        self.tls_ids = network.coordination_graph["agents"]
        # State space.
        # Period is a "Global" state space
        has_period = mdp_params.time_period is not None
        states_depth = None # Assign only when discretize_state_space
        for link in network.coordination_graph["connections"]:
            agent_params_ = deepcopy(agent_params)
            num_phases = mdp_params.phases_per_traffic_light[link[0]] + mdp_params.phases_per_traffic_light[link[1]]
            # Action space.
            if mdp_params.action_space == 'discrete':
                # Discrete action space.
                # In the discrete action space each agent is allowed to pick
                # the signal plan(s) from a set of candidate signal plans given
                # a priori to the system by a user. The candidate signal plans
                # are defined in data/networks/{NETWORK_NAME}/tls_config.json
                actions_depth = mdp_params.num_actions[link[0]] * mdp_params.num_actions[link[1]]
                agent_params_.actions = Bounds(rank=1,
                                               depth=actions_depth)
            else:
                # Continuous action space.
                # In the continuous action space each agent is allowed to select
                # the portion of the cycle length allocated for each of the phases.
                agent_params_.num_phases = num_phases


            # states_rank = num_phases * num_variables + int(has_period)
            states_rank = num_phases * num_variables + 2 * 2
            agent_params_.states = Bounds(states_rank, states_depth)
            # Experience path.
            agent_params_.exp_path = exp_path

            # Name.
            agent_params_.name = link[0] + "_" + link[1]

            # Seed.
            agent_params_.seed = seed

            # Discount factor (gamma).
            agent_params_.discount_factor = mdp_params.discount_factor

            agent = AgentClient(AgentFactory.get(agent_type),
                        params=agent_params_)

            agent.agent1 = link[0]
            agent.agent2 = link[1]

            # Create agent client.
            agents[link[0] + "_" + link[1]] = agent

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
            concat_state = state[agent.agent1] + state[agent.agent2]
            agent.act(concat_state)

        #VE

        # Retrieve requests.
        choices = {tid: "0"
                    for tid in self.tls_ids}

        return choices

    def update(self, s, a, r, s1):
        # Send requests.
        for (tid, agent) in self.agents.items():
            concat_s = s[agent.agent1] + s[agent.agent2]
            concat_s1 = s1[agent.agent1] + s1[agent.agent2]
            summed_r = r[agent.agent1] + r[agent.agent2]

            agent.update(concat_s, a[tid], summed_r, concat_s1)

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
