import numpy as np
import pandas as pd


class ActionTable:
    def __init__(self, agents, data=None):
        self.agents = agents
        self.agent_names = [agent.name for agent in self.agents]

        if len(agents) != 0:
            if data is None:
                size = tuple([len(agent.possible_actions) for agent in self.agents])
                data = np.zeros(size)
            self.table = pd.Series(data.flat,
                                   index=pd.MultiIndex.from_product([agent.possible_actions for agent in self.agents],
                                                                    names=self.agent_names))

    def get_value(self, actions):
        if len(self.agents) == 0:
            return self.action

        res = self.table
        indexer = []
        for agent in self.agent_names:
            indexer.append(actions[agent])
        return res.loc[tuple(indexer)]

    def set_action(self, action):
        self.action = action

    def set_value(self, actions, value):
        res = self.table
        indexer = []
        for agent in self.agent_names:
            indexer.append(actions[agent])

        res.loc[tuple(indexer)] = value
