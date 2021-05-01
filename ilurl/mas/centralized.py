from copy import deepcopy

from ilurl.params import Bounds
from ilurl.loaders.parser import config_parser
from ilurl.agents.client import AgentClient
from ilurl.agents.factory import AgentFactory
from ilurl.interfaces.mas import MASInterface
from functools import reduce
from operator import mul

class CentralizedAgent(MASInterface):
    """
        Centralized multi-agent system wrapper.
    """

    def __init__(self, mdp_params, exp_path, seed):

        # Load agent parameters from config file (train.config).
        agent_type, agent_params = config_parser.parse_agent_params()

        # Create agents.
        agents = {}

        num_variables = len(mdp_params.features)

        # State space.
        # Period is a "Global" state space
        self.has_period = mdp_params.time_period is not None
        states_depth = None # Assign only when discretize_state_space
        agent_params_ = deepcopy(agent_params)
        num_phases = sum(mdp_params.phases_per_traffic_light.values())
        # Action space.
        if mdp_params.action_space == 'discrete':
            # Discrete action space.
            # In the discrete action space each agent is allowed to pick
            # the signal plan(s) from a set of candidate signal plans given
            # a priori to the system by a user. The candidate signal plans
            # are defined in data/networks/{NETWORK_NAME}/tls_config.json
            actions_depth = reduce(mul, mdp_params.num_actions.values(), 1)
            agent_params_.actions = Bounds(rank=1,
                                           depth=actions_depth)
        else:
            # Continuous action space.
            # In the continuous action space each agent is allowed to select
            # the portion of the cycle length allocated for each of the phases.
            agent_params_.num_phases = num_phases

        #TODO: Fix for discretize state space
        # if mdp_params.discretize_state_space:
        #     feature = mdp_params.features[0]
        #     states_depth = len(mdp_params.categories[tid][feature]['0']) + 1

        states_rank = num_phases * num_variables + int(self.has_period)
        agent_params_.states = Bounds(states_rank, states_depth)
        # Experience path.
        agent_params_.exp_path = exp_path

        # Name.
        agent_params_.name = "central_agent"

        # Seed.
        agent_params_.seed = seed

        # Discount factor (gamma).
        agent_params_.discount_factor = mdp_params.discount_factor

        # Create agent client.
        self.agent = AgentClient(AgentFactory.get(agent_type),
                                     params=agent_params_)

    @property
    def stop(self):
        # Send requests.
        self.agent.get_stop()

        # Retrieve requests.
        stops = [self.agent.stop()]

        return all(stops)

    @stop.setter
    def stop(self, stop):
        # Send requests.
        self.agent.set_stop(stop)

        # Synchronize.
        self.agent.receive()

    def act(self, state):
        # Send requests.
        if self.has_period:
            hour = state[next(iter(state))][0]
            concat_state = [e for l in state.values() for e in l[1:]]
            concat_state = tuple([hour] + concat_state)
        else:
            concat_state = tuple([e for l in state.values() for e in l])
        self.agent.act(concat_state)

        return self.agent.receive()

    def update(self, s, a, r, s1):
        # Send requests.
        if self.has_period:
            hour = s[next(iter(s))][0]
            concat_s = [e for l in s.values() for e in l[1:]]
            concat_s = tuple([hour] + concat_s)
        else:
            concat_s = tuple([e for l in s.values() for e in l])

        summed_r = sum(r.values())
        hour = s1[next(iter(s1))][0]
        concat_s1 = [e for l in s.values() for e in l[1:]]
        concat_s1 = tuple([hour] + concat_s1)
        #concat_s1 = tuple([e for l in s1.values() for e in l])
        self.agent.update(concat_s, a, summed_r, concat_s1)

        # Synchronize.
        self.agent.receive()

    def save_checkpoint(self, path):
        # Send requests.
        self.agent.save_checkpoint(path)

        # Synchronize.
        self.agent.receive()

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        # Send requests.
        self.agent.load_checkpoint(chkpts_dir_path, chkpt_num)

        # Synchronize.
        self.agent.receive()

    def terminate(self):
        # Send terminate request for each agent.
        # (also waits for termination)
        self.agent.terminate()
