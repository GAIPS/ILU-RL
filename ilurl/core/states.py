from operator import itemgetter
from collections import defaultdict

import numpy as np

from ilurl.loaders.parser import config_parser
from ilurl.core.rewards import get_rewards

from ilurl.core.meta import (MetaState, MetaStateCollection,
                             MetaStateCategorizer)


def build_states(network, mdp_params, cycle_time):
    """Builder that defines all states

    * States are built upon pre-defined rewards
    * For each tls assigns one or more state objects
    """
    # 1) Handle reward and agent parameters.
    agent_type, _ = config_parser.parse_agent_params()

    rewards, _ = get_rewards()
    if mdp_params.reward not in rewards:
        raise ValueError(f'{mdp_params.reward} not supported')

    # 2) Builds states.
    states = []
    normalize_state_space = mdp_params.normalize_state_space
    tls_max_capacity = network.tls_max_capacity
    phases_per_tls = network.phases_per_tls
    for tid, np in phases_per_tls.items():

        # TODO: Should states be bound directly to rewards
        # 3) For each reward instantiate states.
        if mdp_params.reward == 'MaxSpeedCountReward':
            state_classes = (SpeedState, CountState)
        elif mdp_params.reward == 'MinDelayReward':
            state_classes = (DelayState,)

        for state_cls in state_classes:
            # 4) Define input parameters
            tls_ids = [tid]
            tls_phases = {tid: np}
            max_capacities = {tid: tls_max_capacity[tid]}

            # 5) Q-Learning agent requires function approximation
            # Some objects require discretization
            categorizer = None
            if mdp_params.discretize_state_space:
                if state_cls == SpeedState:
                    categorizer = StateCategorizer(mdp_params.category_speeds)
                elif state_cls == CountState:
                    categorizer = StateCategorizer(mdp_params.category_counts)
                elif state_cls == DelayState:
                    categorizer = StateCategorizer(mdp_params.category_counts)
                else:
                    raise ValueError('Only speed and count states handled')

            # 6) Function specific parameters
            velocity_threshold = None
            if mdp_params.reward == 'MinDelayReward':
                velocity_threshold = mdp_params.velocity_threshold

            state = state_cls(tls_ids, tls_phases,
                              max_capacities, normalize_state_space,
                              categorizer=categorizer,
                              velocity_threshold=velocity_threshold,
                              cycle_time=cycle_time)

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

    def categorize(self):
        # TODO: provide format options: flatten, split 
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
                    digital_state = state.categorize()
                    state_phases.append(int(digital_state[idx][nph]))
                state_tls.append(state_phases)
            ret[tls_id] = state_tls
        return ret


    def flatten(self, values):
        ret = {}
        for tls_id in self.tls_ids:
            tls_obs = values[tls_id]

            flattened = \
                [obs_value for phases in tls_obs for obs_value in phases]

            ret[tls_id] = tuple(flattened)
        return ret
        
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

    def __init__(self, tls_ids, tls_phases, tls_max_caps,
                 normalize, categorizer=None, **kwargs):

        self._tls_ids = tls_ids
        self._tls_phases = tls_phases
        self._tls_caps = {tid: {p: c[1] for p, c in phase_caps.items()}
                          for tid, phase_caps in tls_max_caps.items()
                          if tid in tls_ids}
        self._normalize = False
        if categorizer is not None:
            self.categorizer = categorizer

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

    def reset(self):
        self._memory = {tls_id: {} for tls_id in self.tls_ids}


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
                tls_obs.append(round(val, 2))
            ret.append(tls_obs)

        return ret

    def categorize(self):
        state = self.state
        if hasattr(self, 'categorizer'):
            return self.categorizer.categorize(self.state)


class SpeedState(object, metaclass=MetaState):

    def __init__(self, tls_ids, tls_phases, tls_max_caps,
                 normalize, categorizer=None, **kwargs):

        self._tls_ids = tls_ids
        self._tls_phases = tls_phases
        self._tls_caps = {tid: {p: c[0] for p, c in phase_caps.items()}
                          for tid, phase_caps in tls_max_caps.items()
                          if tid in tls_ids}

        self._normalize = normalize
        
        if categorizer is not None:
            self.categorizer = categorizer

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

    def categorize(self):
        state = self.state
        if hasattr(self, 'categorizer'):
            return self.categorizer.categorize(self.state)
        return self.state


class DelayState(object, metaclass=MetaState):

    def __init__(self, tls_ids, tls_phases, tls_max_caps,
                 normalize, categorizer=None,
                 velocity_threshold=None, cycle_time=None, **kwargs):

        self._tls_ids = tls_ids
        self._tls_phases = tls_phases
        self._tls_caps = {tid: {p: c[0] for p, c in phase_caps.items()}
                          for tid, phase_caps in tls_max_caps.items()
                          if tid in tls_ids}

        self._normalize = normalize
        if categorizer is not None:
            self.categorizer = categorizer

        if velocity_threshold is None:
            raise ValueError('DelayState:velocity_threshold must be provided')
        else:
            self.velocity_threshold = velocity_threshold

        if cycle_time is None:
            raise ValueError('DelayState:cycle_time must be provided')
        else:
            self.cycle_time = cycle_time

        self.reset()

    @property
    def label(self):
        return 'delay'

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

    def update(self, time, veh_ids, veh_speeds, tls_states):
        # 1) New decision point: reset memory
        if time == 0.0:
            self.reset()

        # 2) Update memory with current state
        norm = self._normalize
        for tls_id in self.tls_ids:
            mem = self._memory[tls_id]
            for phase, phase_veh_speeds in veh_speeds[tls_id].items():
                cap = self.tls_caps[tls_id][phase] if norm else 1

                if time not in mem:
                    mem[time] = {}

                # TODO: Empty matrix case nanmean vs 0
                if any(phase_veh_speeds):
                    phase_veh_ids = veh_ids[tls_id][phase]
                    veh_ids_vels = zip(phase_veh_ids, phase_veh_speeds)
                    set_veh_ids = {vid for vid, spd in veh_ids_vels
                                   if (spd / cap) <= self.velocity_threshold}
                    mem[time][phase] = set_veh_ids
                else:
                    mem[time][phase] = set()
            self._memory[tls_id] = mem

    # TODO: create a generator on parent class
    @property
    def state(self):
        # TODO: Add formats
        # TODO: solve the NAN dilemma
        ret = []

        for tls_id, nph in self.tls_phases.items():
            # Get sorted memories by time
            mem = self._memory[tls_id]
            mem = sorted(mem.items(), key=itemgetter(0))
            tls_obs = []
            for p in range(nph):
                # TODO: should discount sim_step
                uniq_veh_ids = set()
                for time, phase_veh_ids in mem:
                    veh_ids = phase_veh_ids[p]
                    uniq_veh_ids = uniq_veh_ids.union(veh_ids)
                tls_obs.append(len(uniq_veh_ids))
            ret.append(tls_obs)
        return ret

    def categorize(self):
        state = self.state
        if hasattr(self, 'categorizer'):
            return self.categorizer.categorize(self.state)
        return self.state


class StateCategorizer(object, metaclass=MetaStateCategorizer):
    # TODO: define upper bound
    def __init__(self, bins):
        self.bins = bins

    def categorize(self, value):
        return np.digitize(value, bins=self.bins)

