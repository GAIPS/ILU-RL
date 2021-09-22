from copy import deepcopy

import numpy as np

from ilurl.mas.ActionTable import ActionTable
from ilurl.params import Bounds
from ilurl.loaders.parser import config_parser
from ilurl.agents.client import AgentClient
from ilurl.agents.factory import AgentFactory
from ilurl.interfaces.mas import MASInterface
from ilurl.mas.VariableElimination import variable_elimination
from ilurl.mas.CGAgent import CGAgent


class CoordinationGraphsMAS(MASInterface):
    """
        Multi-agent system with coordination graphs wrapper.
    """

    def __init__(self, mdp_params, exp_path, seed, network):

        # Load agent parameters from config file (train.config).
        agent_type, agent_params = config_parser.parse_agent_params()
        self.epsilon_init = agent_params.epsilon_init
        self.epsilon_final = agent_params.epsilon_final
        self.epsilon_schedule_timesteps = agent_params.epsilon_schedule_timesteps
        self.current_timestep = 0
        # Create agents.
        self.agents = {}
        for agent in network.coordination_graph["agents"]:
            self.agents[agent] = CGAgent(agent, list(range(mdp_params.num_actions[agent])))

        edges = {}
        num_variables = len(mdp_params.features)
        self.tls_ids = network.coordination_graph["agents"]
        # State space.
        # Period is a "Global" state space
        has_period = mdp_params.time_period is not None
        states_depth = None  # Assign only when discretize_state_space
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

            edge = AgentClient(AgentFactory.get(agent_type),
                               params=agent_params_)

            edge.agent1 = link[0]
            edge.agent2 = link[1]

            qTable = ActionTable([self.agents[link[0]], self.agents[link[1]]])

            # Create agent client.
            edges[link[0] + "_" + link[1]] = edge

            self.agents[link[0]].edges.append(edge)
            self.agents[link[0]].dependant_agents.append(link[1])
            self.agents[link[0]].payout_functions.append(qTable)

            self.agents[link[1]].edges.append(edge)
            self.agents[link[1]].dependant_agents.append(link[0])
            self.agents[link[1]].payout_functions.append(qTable)

        self.edges = edges


    @property
    def stop(self):
        # Send requests.
        for (tid, agent) in self.edges.items():
            agent.get_stop()

        # Retrieve requests.
        stops = [agent.stop for (_, agent) in self.edges.items()]

        return all(stops)

    @stop.setter
    def stop(self, stop):
        # Send requests.
        for (tid, agent) in self.edges.items():
            agent.set_stop(stop)

        # Synchronize.
        for (tid, agent) in self.edges.items():
            agent.receive()

    def act(self, state, test=False):
        # Send requests.
        #print("\nQTables:")

        for (tid, agent) in self.edges.items():
            concat_state = state[agent.agent1] + state[agent.agent2]
            agent.act(concat_state)
            agent.receive()

            qTable = self.forward_pass(tid, concat_state)
            #print(tid)
            #print(qTable[tid][0:2])
            #print(qTable[tid][2:4])

            agent1 = self.agents[self.edges[tid].agent1]
            agent2 = self.agents[self.edges[tid].agent2]
            payoutFunction = ActionTable([agent1, agent2], qTable[tid])
            # agent1.qtable = payoutFunction
            agent1.payout_functions = [payoutFunction]
            agent2.payout_functions = [payoutFunction]


        self.current_timestep += 5
        # VE
        choices = variable_elimination(self.agents, epsilon=self.getEpsilon(),  test=test)
        #print("\nActions: ", sorted(choices.items()))


        return choices

    def tuple_to_int_actions(self, actions):

        return {tid: actions[agent.agent1] * 2 ** 1 + actions[agent.agent2] * 2 ** 0
                for (tid, agent) in self.edges.items()}

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def getEpsilon(self):
        return self.epsilon_init - ((self.epsilon_init - self.epsilon_final) * (
                    self.current_timestep / self.epsilon_schedule_timesteps))

    def forward_pass(self, tid, state):

        self.edges[tid].forward_pass(state)

        return {tid: self.softmax(self.edges[tid].receive()[0])}

    def update(self, s, a, r, s1):
        # Send requests.

        a = self.tuple_to_int_actions(a)
        for (tid, agent) in self.edges.items():
            concat_s = s[agent.agent1] + s[agent.agent2]
            concat_s1 = s1[agent.agent1] + s1[agent.agent2]
            summed_r = r[agent.agent1] + r[agent.agent2]
            #print("\nRewards: Agent1: %f, Agent2: %f, Sum: %f" % (r[agent.agent1], r[agent.agent2], summed_r))
            agent.update(concat_s, a[tid], summed_r, concat_s1)

        # Synchronize.
        for (tid, agent) in self.edges.items():
            agent.receive()


    def save_checkpoint(self, path):
        # Send requests.
        for (tid, agent) in self.edges.items():
            agent.save_checkpoint(path)

        # Synchronize.
        for (tid, agent) in self.edges.items():
            agent.receive()

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        # Send requests.
        for (tid, agent) in self.edges.items():
            agent.load_checkpoint(chkpts_dir_path, chkpt_num)

        # Synchronize.
        for (tid, agent) in self.edges.items():
            agent.receive()

    def terminate(self):
        # Send terminate request for each agent.
        # (also waits for termination)
        for (tid, agent) in self.edges.items():
            agent.terminate()
