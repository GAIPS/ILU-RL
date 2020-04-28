from copy import deepcopy

import numpy as np

from ilurl.core.params import Bounds
import ilurl.loaders.parsers as parsers

from ilurl.core.ql.agent import QL
from ilurl.core.dqn.agent import DQN

AGENT_TYPES = ('QL', 'DQN')

class AgentsWrapper(object):

    def __init__(self, 
                mdp_params,
                agent_type):

        if agent_type not in AGENT_TYPES:
            raise ValueError(f'''
                Agent type must be in {AGENT_TYPES}.
                Got {agent_type} type instead.''')

        # Load agent parameters from config file.
        agent_params = parsers.parse_agent_params(agent_type)

        # Create agents.
        agents = {}

        num_variables = len(mdp_params.states_labels)

        # TODO: Afterwards this needs to come from a config file 
        # telling what each agent is controlling.
        # TODO: assumes each agent controls 1 intersection
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

    def act(self, state):

        choices = {}

        for tid in self.agents.keys():
            agent = self.agents[tid]
            choices[tid] = agent.act(state[tid])

        return choices

    def update(self, s, a, r, s1):

        for tid in self.agents.keys():
            agent = self.agents[tid]
            s_, a_, r_, s1_ = s[tid], a[tid], r[tid], s1[tid]
            agent.update(s_, a_, r_, s1_)

    """ @property
    def Q(self):
        self._Q = {i: _QL_agent.Q
                   for i, _QL_agent in enumerate(self._QL_agents)}

        return self._Q

    @Q.setter
    def Q(self, Q):
        for i, Qi in Q.items():
              self._QL_agents[i].Q = Qi

    @property
    def explored(self):
        self._explored = {i: _QL_agent.explored
                   for i, _QL_agent in enumerate(self._QL_agents)}
        return self._explored

    @property
    def visited_states(self):
        self._visited_states = {i: _QL_agent.visited_states
                   for i, _QL_agent in enumerate(self._QL_agents)}
        return self._visited_states

    @property
    def Q_distances(self):
        self._Q_distances = {i: _QL_agent.Q_distances
                   for i, _QL_agent in enumerate(self._QL_agents)}
        return self._Q_distances

    @property
    def stop(self):
        stops = [_QL_agent.stop for _QL_agent in self._QL_agents]
        return all(stops)

    @stop.setter
    def stop(self, stop):
        for _QL_agent in self._QL_agents:
            _QL_agent.stop = stop
        return stop """
