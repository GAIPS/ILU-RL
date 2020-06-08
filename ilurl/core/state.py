import pdb
from collections import defaultdict

import numpy as np


from ilurl.loaders.parser import config_parser
from ilurl.core.rewards import get_rewards
from ilurl.core.meta import (MetaState, MetaStateCollection,
                              MetaStateCalculator)
# import ilurl.core.rewards as rw

# TODO: check if the module has one of the following names
# REWARDS = ['MaxSpeedCountReward', 'MinDelayReward', 'MinDeltaDelayReward']


def build_states(network, mdp_params):
    """Builder that defines all states

    * States are built upon pre-defined rewards
    * For each tls assigns one or more state objects
    """
    # 1) Handle reward and agent parameters.
    reward_type = mdp_params.reward
    agent_type, _ = config_parser.parse_agent_params()

    rewards, _ = get_rewards()
    if mdp_params.reward not in rewards:
        raise ValueError(f'{mdp_params.reward} not supported')

    # TODO: test agents

    # 2) Builds states.
    states = []
    normalize_state_space = mdp_params.normalize_state_space
    tls_max_capacity = network.tls_max_capacity
    phases_per_tls = network.phases_per_tls
    for tid, np in phases_per_tls.items():

        # 3) For each reward instantiate states.
        if reward_type == 'MaxSpeedCountReward':
            state_classes = (SpeedState, CountState)

        for state_class in state_classes:
            tls_ids = [tid]
            tls_phases = {tid:np}
            max_capacities = {tid: tls_max_capacity[tid]}
            state = state_class(tls_ids, tls_phases,
                                max_capacities, normalize_state_space)

            # 4) Q-Learning agent requires function approximation
            if agent_type == 'QL':

                # TODO: extend MDP params normalize per state
                # normalize = state.label == 'speed'

                # category_key = f'category_{state.label}s'
                # category_bins = getattr(mdp_params, category_key, None)
                # if category_bins is None:
                #     raise ValueError(
                #         'Category bins for `QL` must be defined'
                #     )

                # # TODO: extend MDP params for max_speed (network)
                # category_max = f'max_{state.label}'
                # if normalize:
                #     ubound = 1.0
                # else:
                #     # either 30 vehs or 30 m/s
                #     ubound = 30
                # category_bins = getattr(mdp_params, category_max, ubound)

                # categorizer = StateCategorizer(
                #     category_bins,
                #     category_max,
                #     normalize
                # )

                # state.calculator = categorizer
                pass
            states.append(state)
    return StateCollection(states)


class StateCollection(object, metaclass=MetaStateCollection):

    def __init__(self, states):
        self.tls_ids = \
            sorted({tid for s in states for tid in s.tls_ids})
        
        self.tls_phases = \
            {tls_id: state.tls_phases[tls_id]
             for tls_id in self.tls_ids
             for state in states if tls_id in state.tls_ids}

        self._states = states
        
    @property
    def label(self):
        labels = []
        for tls_id in self.tls_ids:
            states = [s for s in self._states if tls_id in s.tls_ids]
            for state in states:
                if state.label not in labels:
                    labels.append(state.label)
        return '|'.join(labels)

    @property
    def tls_ids(self):
        return self._tls_ids

    @tls_ids.setter
    def tls_ids(self, tls_ids):
        self._tls_ids = tls_ids

    @property
    def tls_phases(self):
        return self._tls_phases

    @tls_phases.setter
    def tls_phases(self, tls_phases):
        self._tls_phases = tls_phases

    def update(self, time, veh_ids, veh_speeds, tls_states):

        for tls_id in self.tls_ids:
            if tls_id in veh_ids and tls_id in veh_speeds:
                for state in self._states:
                    if tls_id in state.tls_ids:
                        state.update(time,
                                     veh_ids,
                                     veh_speeds,
                                     tls_states)

    def reset(self):
        for state in self._states:
            state.reset()

    @property
    def state(self):
        # TODO: provide format options: flatten, split,   
        ret = {}
        for tls_id in self.tls_ids:
            states = [s for s in self._states if tls_id in s.tls_ids]
            num_phases = self.tls_phases[tls_id]
            state_tls = []
            for nph in range(num_phases):
                state_phases = []
                for state in states:
                    # Convers the case of multiple tls
                    idx = state.tls_ids.index(tls_id)
                    state_phases.append(state.state[idx][nph])
                state_tls.append(state_phases)
            ret[tls_id] = state_tls
        return ret

    def calculate(self):
        nested_states = \
                [s.calculate() for tid in self.tls_ids
                 for s in self._states if tid in s.tls_ids]

        return [s for states in nested_states for s in states]

    def split(self):
        """Splits per state label"""
        labels = self.label.split('|')
        ret = {} # defaultdict(list)
        for tls_id in self.tls_ids:
            ret[tls_id] = defaultdict(list)
            state = self.state[tls_id]
            for nph in range(self.tls_phases[tls_id]):
                for idx, label in enumerate(labels):
                    ret[tls_id][label].append(state[nph][idx])

        ret = {tls_id: tuple([v for v in data.values()])
               for tls_id, data in ret.items()}
        return ret

class CountState(object, metaclass=MetaState):

    def __init__(self, tls_ids, tls_phases, tls_max_caps, normalize):
        self._tls_ids = tls_ids
        self._tls_phases = tls_phases
        self._tls_caps = {tid: {p: c[1] for p, c in phase_caps.items()}
                          for tid, phase_caps in tls_max_caps.items()
                          if tid in tls_ids}
        self._normalize = False
        self.reset()

    @property
    def label(self):
        return 'count'

    @property
    def tls_ids(self):
        return self._tls_ids

    @property
    def tls_phases(self):
        return self._tls_phases

    @property
    def tls_caps(self):
        return self._tls_caps

    def update(self, time, veh_ids, veh_speeds, tls_states):


        # TODO: context manager
        for tls_id in self.tls_ids:
            mem = self._memory[tls_id]
            for phase, phase_veh_ids in veh_ids[tls_id].items():
                if time not in mem:
                    mem[time] = {}
                mem[time][phase] = len(phase_veh_ids)
            self._memory[tls_id] = mem
        # if time not in self._memory:
        #     self._memory[time] = {}

        # for phase, phase_veh_ids in veh_ids.items():
        #     self._memory[time][phase] = len(phase_veh_ids)

    def reset(self):
        self._memory = {tls_id: {} for tls_id in self.tls_ids}


    # TODO: create a generator on parent class
    # @property
    # def state(self):
    #     # ret = 0
    #     # if self._memory:
    #     # TODO: Add flatten as option
    #     # ret = {p: np.nanmean([v for v in self._memory[t][p]])
    #     #        for t in self._memory
    #     #        for p in range(self.tls_phases)}

    #     ret = []
    #     # ret = defaultdict(list)
    #     for tls_id, nph in self.tls_phases.items():
    #         mem = self._memory[tls_id]
    #         tls_obs = []
    #         for p in range(nph):
    #             vals = [pv[p] for pv in mem.values()]
    #             # TODO: solve the NAN dilemma
    #             val = np.nanmean(vals)
    #             val = 0.0 if np.isnan(val) else val
    #             tls_obs.append(val)
    #         ret.append(val)
    #     #TODO: include format options

    # TODO: create a generator on parent class
    @property
    def state(self):
        # TODO: Add formats
        # TODO: solve the NAN dilemma
        ret = []
        for tls_id, nph in self.tls_phases.items():
            mem = self._memory[tls_id]
            tls_obs = []
            for p in range(nph):
                vals = [pv[p] for pv in mem.values()]
                val = np.nanmean(vals) if vals else 0.0
                # val = 0.0 if np.isnan(val) else val
                tls_obs.append(round(val, 2))
            ret.append(tls_obs)

        return ret

    def calculate(self):
        if hasattr(self, 'calculator'):
            return self.calculator.calculate(self.state)
        return self.state


class SpeedState(object, metaclass=MetaState):

    def __init__(self, tls_ids, tls_phases, tls_max_caps, normalize):
        self._tls_ids = tls_ids
        self._tls_phases = tls_phases
        self._tls_caps = {tid: {p: c[0] for p, c in phase_caps.items()}
                          for tid, phase_caps in tls_max_caps.items()
                          if tid in tls_ids}

        self._normalize = normalize
        self.reset()

    @property
    def label(self):
        return 'speed'

    @property
    def tls_ids(self):
        return self._tls_ids

    @property
    def tls_phases(self):
        return self._tls_phases

    @property
    def tls_caps(self):
        return self._tls_caps

    def reset(self):
        self._memory = {tls_id: {} for tls_id in self.tls_ids}

    # TODO: create a generator on parent class
    @property
    def state(self):
        # TODO: Add formats
        # TODO: solve the NAN dilemma
        ret = []
        norm = self._normalize
        for tls_id, nph in self.tls_phases.items():
            mem = self._memory[tls_id]
            tls_obs = []
            for p in range(nph):
                cap = self.tls_caps[tls_id][p] if norm else 1

                vals = [pv[p] for pv in mem.values()]
                val = np.nanmean(vals) if any(vals) else 0.0
                tls_obs.append(round(val / cap, 2))
            ret.append(tls_obs)

        return ret

    def update(self, time, veh_ids, veh_speeds, tls_states):

        # TODO: Add mem context manager
        for tls_id in self.tls_ids:
            mem = self._memory[tls_id]
            for phase, phase_veh_speeds in veh_speeds[tls_id].items():
                if time not in mem:
                    mem[time] = {}

                # TODO: Empty matrix case nanmean vs 0
                if any(phase_veh_speeds):
                    mem[time][phase] = \
                        round(np.nanmean([pvs for pvs in phase_veh_speeds]), 2)
                else:
                    mem[time][phase] = 0.0
            self._memory[tls_id] = mem

    def calculate(self):
        if hasattr(self, 'calculator'):
            return self.calculator.calculate(self.state)
        return self.state

class StateCategorizer(object, metaclass=MetaStateCalculator):
    # TODO: define upper bound
    def __init__(self, bins, ubound, normalize):
        self.bins = bins
        self.ubound = ubound
        self.normalize = normalize

    def calculate(self):
        return {p: self._calculate(v)
                for p, v in self.state.items()}

    def _calculate(self, value):
        if normalize:
            value = value / self.ubound
        return np.digitalize(value, bins=self.bins)

