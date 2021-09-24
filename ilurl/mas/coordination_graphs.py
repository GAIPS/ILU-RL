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
        num_variables = len(mdp_params.features)
        states_depth = None # Assign only when discretize_state_space

        self.epsilon_init = agent_params.epsilon_init
        self.epsilon_final = agent_params.epsilon_final
        self.epsilon_schedule_timesteps = agent_params.epsilon_schedule_timesteps
        self.current_timestep = 0

        # Create agents.
        self.agents = {}
        for tid in mdp_params.phases_per_traffic_light:
            self.agents[tid] = CGAgent(tid, list(range(mdp_params.num_actions[tid])))

        tid = list(mdp_params.phases_per_traffic_light)[0]
        agent_params_ = deepcopy(agent_params)
        num_phases = mdp_params.phases_per_traffic_light[tid]
        actions_depth = mdp_params.num_actions[tid]
        agent_params_.actions = Bounds(rank=1,
                                       depth=actions_depth)
        states_rank = num_phases * num_variables + 2
        agent_params_.states = Bounds(states_rank, states_depth)
        agent_params_.exp_path = exp_path
        agent_params_.name = "global_node_agent"
        agent_params_.seed = seed
        agent_params_.discount_factor = mdp_params.discount_factor
        self.node_dqn_agent = AgentClient(AgentFactory.get(agent_type),
                                      params=agent_params_)

        num_variables = len(mdp_params.features)
        self.tls_ids = network.coordination_graph["agents"]
        # State space.
        # Period is a "Global" state space
        has_period = mdp_params.time_period is not None
        states_depth = None  # Assign only when discretize_state_space

        link = network.coordination_graph["connections"][0]
        agent_params_ = deepcopy(agent_params)
        num_phases = mdp_params.phases_per_traffic_light[link[0]] + mdp_params.phases_per_traffic_light[link[1]]
        actions_depth = mdp_params.num_actions[link[0]] * mdp_params.num_actions[link[1]]
        agent_params_.actions = Bounds(rank=1,
                                       depth=actions_depth)

        states_rank = num_phases * num_variables + 2 * 2
        agent_params_.states = Bounds(states_rank, states_depth)
        agent_params_.exp_path = exp_path

        agent_params_.name = "global_edge_agent"
        agent_params_.seed = seed
        agent_params_.discount_factor = mdp_params.discount_factor

        self.edge_dqn_agent = AgentClient(AgentFactory.get(agent_type),
                                          params=agent_params_)

        for link in network.coordination_graph["connections"]:

            self.agents[link[0]].dependant_agents.append(link[1])
            self.agents[link[1]].dependant_agents.append(link[0])

        self.edges = network.coordination_graph["connections"]


    @property
    def stop(self):
        self.edge_dqn_agent.get_stop()
        stop = self.edge_dqn_agent.stop
        # stops = [agent.stop for (_, agent) in self.edges.items()]
        return stop

    @stop.setter
    def stop(self, stop):
        self.edge_dqn_agent.set_stop(stop)
        self.edge_dqn_agent.receive()

    def act(self, state, test=False):
        # Send requests.
        #print("\nQTables:")

        for (tid, agent) in  self.agents.items():
            qTable = self.forward_pass(self.node_dqn_agent, state[tid])
            payoutFunction = ActionTable([self.agents[tid]], qTable)
            agent.payout_functions = [payoutFunction]
            self.node_dqn_agent.act(state[tid])
            self.node_dqn_agent.receive()
            #print(tid)
            #print(qTable[0:2])


        for (agent1, agent2) in self.edges:
            concat_state = state[agent1] + state[agent2]
            self.edge_dqn_agent.act(concat_state)
            self.edge_dqn_agent.receive()

            qTable = self.forward_pass(self.edge_dqn_agent, concat_state)
            #print(agent1 + "_" + agent2)
            #print(qTable[0:2])
            #print(qTable[2:4])

            agent1 = self.agents[agent1]
            agent2 = self.agents[agent2]
            payoutFunction = ActionTable([agent1, agent2], qTable)
            agent1.payout_functions.append(payoutFunction)
            agent2.payout_functions.append(payoutFunction)


        self.current_timestep += 5
        # VE
        choices = variable_elimination(self.agents, epsilon=self.getEpsilon(),  test=test)
        #print("\nActions: ", sorted(choices.items()))


        return choices

    def tuple_to_int_actions(self, action1, action2):

        return action1 * 2 ** 1 + action2 * 2 ** 0


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def getEpsilon(self):
        return self.epsilon_init - ((self.epsilon_init - self.epsilon_final) * (
                    self.current_timestep / self.epsilon_schedule_timesteps))

    def forward_pass(self, agent, state):
        agent.forward_pass(state)
        return self.softmax(agent.receive()[0])

    def update(self, s, a, r, s1):
        # Send requests.

        for (tid, agent) in self.agents.items():
            self.node_dqn_agent.update(s[tid], a[tid], r[tid], s1[tid])
            self.node_dqn_agent.receive()
            #print("\nRewards: %s: %f" % (tid, r[tid]))

        for (agent1, agent2) in self.edges:
            concat_s = s[agent1] + s[agent2]
            concat_s1 = s1[agent1] + s1[agent2]
            #summed_r = sum(r.values())
            summed_r = r[agent1] + r[agent2]
            action = self.tuple_to_int_actions(a[agent1], a[agent2])
            self.edge_dqn_agent.update(concat_s, action, summed_r, concat_s1)
            self.edge_dqn_agent.receive()

    def save_checkpoint(self, path):
        self.edge_dqn_agent.save_checkpoint(path)
        self.edge_dqn_agent.receive()

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        self.edge_dqn_agent.load_checkpoint(chkpts_dir_path, chkpt_num)
        self.edge_dqn_agent.receive()

    def terminate(self):
        # Send terminate request for the agent.
        # (also waits for termination)
        self.edge_dqn_agent.terminate()
