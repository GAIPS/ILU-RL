"""Implementation of states as perceived by agent"""
import re
import inspect
from sys import modules
from operator import itemgetter
from collections import defaultdict
from copy import deepcopy


import numpy as np

from ilurl.loaders.parser import config_parser
from ilurl.meta import (MetaState, MetaStateCollection,
                             MetaStateCategorizer)
from ilurl.utils.aux import camelize, flatten



def get_states():
    """States defined within module

    * Uses module introspection to get the handle for classes
    * Ignores StateCollection, StateCategorizer


    Returns:
    -------
    * names: tuple(<str>)
        camelized names of state like classes defined within this module

    * objects: tuple(<objects>)
        classes wrt camelized names

    Usage:
    -----
    > names, objs = get_states()
    > names
    > ('count', 'delay', 'speed')
    > objs
    > (<class 'ilurl.core.states.CountState'>,
       <class 'ilurl.core.states.DelayState'>,
       <class 'ilurl.core.states.SpeedState'>)
    """
    this = modules[__name__]
    names, objects = [], []
    non_states = ('StateCategorizer', 'StateCollection', 'LagState')
    for name, obj in inspect.getmembers(this):

        # Is a definition a class?
        if inspect.isclass(obj):
            if  name not in non_states:
                # Is defined in this module
                if inspect.getmodule(obj) == this:
                    name = camelize(name.replace('State', ''))
                    names.append(name)
                    objects.append(obj)

    return tuple(names), tuple(objects)


def build_states(network, mdp_params):
    """Builder that defines all states

    * States are built upon pre-defined rewards
    * For each tls assigns one or more state objects

    Params:
    ------

    * network: ilurl.networks.base.Network
        network to be described by states

    * mdp_params: ilurl.core.params.MDPParams
        mdp specification: agent, states, rewards, gamma and learning params

    Returns:
    --------

    * state_collection: ilurl.core.states.StateCollection
        a wrapper for multiple states
    """
    # 1) Handle state, reward and agent parameters.
    states_names, states_classes = get_states()
    req_states_names = {s for s in mdp_params.states if 'lag' not in s}
    no_state_set = req_states_names - set(states_names)
    if len(no_state_set) > 0:
        raise ValueError(f'{no_state_set} is not implemented')

    # 2) Builds states.
    states = []
    normalize_state_space = mdp_params.normalize_state_space
    tls_max_capacity = network.tls_max_capacity
    phases_per_tls = network.phases_per_tls

    matcher = re.compile('\[(.*?)\]')

    # Global states.
    time_period = mdp_params.time_period

    for tid, np in phases_per_tls.items():

        for state_name in mdp_params.states:
            # TODO: Necessary to create base state first
            if 'lag' in state_name:
                # derived state must have been already created.
                derived_state_tuple = matcher.search(state_name).groups(0)
                derived_state = \
                    [s for s in states
                     if s.label in derived_state_tuple and
                     tid in s.tls_ids][0]

                state = LagState(derived_state)
            else:

                state_cls = states_classes[states_names.index(state_name)]

                # 3) Define input parameters.
                tls_ids = [tid]
                tls_phases = {tid: np}
                max_capacities = {tid: tls_max_capacity[tid]}

                # 4) Q-Learning agent requires function approximation
                # Some objects require discretization
                categorizer = None
                if mdp_params.discretize_state_space:
                    if state_cls == SpeedState:
                        categorizer = StateCategorizer(mdp_params.category_speeds)
                    elif state_cls == CountState:
                        categorizer = StateCategorizer(mdp_params.category_counts)
                    elif state_cls == DelayState:
                        categorizer = StateCategorizer(mdp_params.category_delays)
                    elif state_cls == QueueState:
                        categorizer = StateCategorizer(mdp_params.category_queues)
                    else:
                        raise ValueError(f'No discretization bins for {state_cls}')

                # 5) Function specific parameters
                velocity_threshold = None
                velocity_threshold = mdp_params.velocity_threshold

                state = state_cls(tls_ids, tls_phases,
                                  max_capacities, normalize_state_space,
                                  categorizer=categorizer,
                                  velocity_threshold=velocity_threshold)

            states.append(state)
    return StateCollection(states, time_period)


class StateCollection(object, metaclass=MetaStateCollection):

    def __init__(self, states, time_period=None):
        # Time is a GlobalState
        self._time_state = TimeState(time_period) if time_period else None

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

        if self._time_state:
            labels.append(self._time_state.label)

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

    def update(self, time, vehs, tls=None):

        if self._time_state:
            self._time_state.update(time)

        for tls_id in self.tls_ids:
            if tls_id in vehs:
                for state in self._states:
                    if tls_id in state.tls_ids:
                        state.update(time, vehs, tls)

    def reset(self):
        if self._time_state:
            self._time_state.reset()

        for state in self._states:
            state.reset()

    def state(self, filter_by=None):
        # TODO: provide format options: flatten, split,   
        ret = {}
        for tls_id in self.tls_ids:
            states = [s for s in self._states if tls_id in s.tls_ids]
            if filter_by is not None:
                states = [s for s in states if s.label in filter_by]

            has_time = (filter_by and 'time' in filter_by) or None
            num_phases = self.tls_phases[tls_id]
            state_tls = [self._time_state.state] if has_time else []
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

        has_time = self._time_state is not None
        for tls_id in self.tls_ids:
            states = [s for s in self._states if tls_id in s.tls_ids]
            num_phases = self.tls_phases[tls_id]
            state_tls = [self._time_state.state] if has_time else []
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
            ret[tls_id] = tuple(flatten(values[tls_id]))
        return ret

    def split(self, filter_by=None):
        """Splits per state label"""
        labels = self.label.split('|')
        ret = {}
        state = self.state(filter_by=filter_by)
        if filter_by:
            labels = filter_by

        for tls_id in self.tls_ids:
            ret[tls_id] = defaultdict(list)
            _state = state[tls_id]
            for nph in range(self.tls_phases[tls_id]):
                for idx, label in enumerate(labels):
                    ret[tls_id][label].append(_state[nph][idx])


        has_time = (filter_by and 'time' in filter_by) or None
        time_list = [self._time_state.state] if has_time else [] 
        ret = {tls_id: tuple(time_list + [v for v in data.values()])
               for tls_id, data in ret.items()}
        return ret


class QueueState(object, metaclass=MetaState):
    """Measures the max queue wrt to each lane on a phase,
        over a cycle.

    Let k be decision step and d elapsed time, after
       decision k was made (duration), t is the time and
       duration is t - tk.

    The maximum queue is taken over all lanes that belong
    to the lane group corresponding phase i, Li. Averaged
    over duration (t-tk).

    s^k_i = mean{max{q^d_l, l in Li}, d in [0:cycle_time - 1]}

    where:
       * i phase
       * l in Li for all i = {1, 2, 3, ...N}
       * k is the decision step.
       * q^d_l is the number of queued vehicles
               in lane l at duration t.

       q^d_l = len([veh for veh in V^t_l if vel < thresh])

    where:
       * V^t_l the set of all veh_ids on lane l
               at time t.

    * Equivalent to state definition 1: Queue Length

     Reference:
    ----------
    * El-Tantawy, et al. 2014
        "Design for Reinforcement Learning Parameters for Seamless"

    See also:
    --------
    * Balaji et al., 2010
        "Urban traffic signal control using reinforcement learning agent"

    * Richter, S., Aberdeen, D., & Yu, J., 2007
        "Natural actor-critic for road traffic optimisation."

    * Abdulhai et al., 2003
        "Reinforcement learning for true adaptive traffic signal
        control."
    """



    def __init__(self, tls_ids, tls_phases, tls_max_caps,
                 normalize, categorizer=None, velocity_threshold=None,
                 **kwargs):

        self._tls_ids = tls_ids
        self._tls_phases = tls_phases
        self._tls_caps = {tid: {p: c[1] for p, c in phase_caps.items()}
                          for tid, phase_caps in tls_max_caps.items()
                          if tid in tls_ids}

        self._normalize = normalize
        if velocity_threshold is None:
            raise ValueError(
                f'QueueState: velocity_threshold must be provided')
        else:
            self.velocity_threshold = velocity_threshold

        if categorizer is not None:
            self.categorizer = categorizer

        self.reset()

    @property
    def label(self):
        return 'queue'

    @property
    def tls_ids(self):
        return self._tls_ids

    @property
    def tls_phases(self):
        return self._tls_phases

    @property
    def tls_caps(self):
        return self._tls_caps

    def update(self, time, vehs, tls=None):
        """Q^k_l the set of all veh_ids on lane l decision step k.

            * Computes V^k_l the set of all veh_ids on lane l at time k.
            * Every car moving below velocity_threshold is
                considered to be enqueued.

            * Stores for each time, the ids for the enqueued
                vehicles.


            * Computes V^k_l the set of all veh_ids on lane l at time k.

                q^k_l = len([veh for veh in V^k_l if ind(veh)])

        """
        # TODO: Add edges

        # 2) Update memory with current state.
        # TODO include veh_edges
        norm = self._normalize
        thresh = self.velocity_threshold
        for tls_id in self.tls_ids:

            # 1) New decision point: reset
            if time == 0.0:
                self.reset(tls_id)

            mem = self._memory[tls_id]

            for phase, _vehs in vehs[tls_id].items():
                cap = self.tls_caps[tls_id][phase] if norm else 1

                _vehs_lanes = [_veh.lane for _veh in _vehs]
                _vehs_speeds = [_veh.speed for _veh in _vehs]
                _vehs_ids = [_veh.id for _veh in _vehs]

                uniq_veh_lanes = sorted(set(_vehs_lanes))
                if time not in mem:
                    mem[time] = {}

                if any(_vehs_ids):
                    _triplet = zip(_vehs_lanes, _vehs_speeds, _vehs_ids)

                    queues = {
                        lane: [vh for ln, vl, vh in _triplet
                               if ln == lane and (vl / cap) <= thresh]
                        for lane in uniq_veh_lanes
                    }

                else:
                    queues = {lane: [] for lane in uniq_veh_lanes}

                mem[time][phase] = queues

            self._memory[tls_id] = mem

    def reset(self, tls_id=None):
        """Memory reset

        Params:
        ------
        * tls_id .: str
            traffic light signal id if None reset all

        """
        def fn(x):
            return tls_id is None or tls_id == x

        mem = {tid: {} for tid in self.tls_ids if fn(tid)}
        self._memory = mem

    @property
    def state(self):
        ret = []
        # 1) Decision points data should be independent.
        for tls_id, nph in self.tls_phases.items():
            mem = self._memory[tls_id]
            tls_ret = []
            # 2) Gets the recent observations
            for duration, data in mem.items():
                tls_obs = []

                for phase in range(nph):
                    delta = 0
                    phase_data = data[phase]

                    for veh_ids in phase_data.values():
                        set_veh_ids = set(veh_ids)
                        delta = max(len(set_veh_ids), delta)
                    tls_obs.append(delta)
                tls_ret.append(tls_obs)
            ret.append(np.mean(np.array(tls_ret), axis=0).tolist())
        return ret

    def categorize(self):
        state = self.state
        if hasattr(self, 'categorizer'):
            return self.categorizer.categorize(self.state)

class TimeState(object):
    """Keeps track of time"""

    def __init__(self, period, **kwargs):
        self._period = period
        self.reset()

    @property
    def label(self):
        return 'time'

    def update(self, time, vehs=None, tls=None):
        # prevents update from being called multiple times
        if time != self._last:
            self._last = time
            self._memory += 1


    def reset(self):
        self._last = -1
        self._memory = -1

    @property
    def state(self):
        return int(self._memory / self._period)

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

    def update(self, time, vehs, tls=None):

        # TODO: context manager
        for tls_id in self.tls_ids:
            mem = self._memory[tls_id]
            for phase, _veh_ids in vehs[tls_id].items():
                if time not in mem:
                    mem[time] = {}
                mem[time][phase] = len(_veh_ids)
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


    def update(self, time, vehs, tls=None):
        # TODO: Add mem context manager
        for tls_id in self.tls_ids:
            mem = self._memory[tls_id]
            for phase, _vehs in vehs[tls_id].items():
                if time not in mem:
                    mem[time] = {}

                _veh_speeds = [_veh.speed for _veh in _vehs]
                # TODO: Empty matrix case nanmean vs 0
                if any(_veh_speeds):
                    mem[time][phase] = \
                        round(np.nanmean([pvs for pvs in _veh_speeds]), 2)
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



class LagState(object, metaclass=MetaState):
    """Retains the previous cycle state's value

    This is a special kind of state which is derived from
    some other state -- it just keeps tabs on which was
    the last cycle's value.

    Uses delegation to register a common interface with
    the other state instances.

    """

    def __init__(self, state, lag=1):
        self._state = state
        self._state_0 = [[0] * self.tls_phases[tid] for tid in self.tls_ids]
        self._num_cycles = -1
        self._lag = lag
        self.reset()

    @property
    def label(self):
        return f'lag[{self._state.label}]'

    @property
    def tls_ids(self):
        return self._state.tls_ids

    @property
    def tls_phases(self):
        return self._state.tls_phases

    @property
    def tls_caps(self):
        return self._state.tls_caps

    def reset(self):
        self._memory = {}

    def update(self, time, vehs, tls=None):
        # 1) New decision point: keeps a rolling state
        if time == 0.0:
            self._num_cycles += 1
            keys = [k for k in self._memory if k >= 0 and
                    self._num_cycles - k > self._lag]
            for k in keys:
                del self._memory[k]

        # 2) Update memory with derived state
        # TODO: How to know the state is updated?
        # Option 1: `violate' constraint and check previous time
        # Option 2: `dumb' redo work
        _state = deepcopy(self._state)
        _state.update(time, vehs, tls=tls)
        self._memory[self._num_cycles] = _state

    # TODO: create a generator on parent class
    @property
    def state(self):
        k = self._num_cycles - self._lag
        if k in self._memory:
            return self._memory[k].state
        return self._state.state

    def categorize(self):
        k = self._num_cycles - self._lag
        state = self._memory.get(k, self._state_0)
        if hasattr(state, 'categorizer'):
            return state.categorize()
        return state


class DelayState(object, metaclass=MetaState):
    """Computes the total delay observed per phase

    References:
    ----------
    * El-Tantawy, et al. 2014
        "Design for Reinforcement Learning Parameters for Seamless"

    See also:
    --------
   
    * Lu, Liu, & Dai. 2008
        "Incremental multistep Q-learning for adaptive traffic signal control"

    * Shoufeng et al., 2008
        "Q-Learning for adaptive traffic signal control based on delay"

    * Abdullhai et al. 2003
        "Reinforcement learning for true adaptive traffic signal control."

    * Wiering, 2000
        "Multi-agent reinforcement learning for traffic light control."
    """

    def __init__(self, tls_ids, tls_phases, tls_max_caps,
                 normalize, categorizer=None,
                 velocity_threshold=None, **kwargs):

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

    def update(self, time, vehs, tls=None):
        # 1) New decision point: reset memory
        if time == 0.0:
            self.reset()

        # 2) Update memory with current state
        norm = self._normalize
        for tls_id in self.tls_ids:
            mem = self._memory[tls_id]
            for phase, _vehs in vehs[tls_id].items():
                cap = self.tls_caps[tls_id][phase] if norm else 1

                if time not in mem:
                    mem[time] = {}

                # TODO: Empty matrix case nanmean vs 0
                if any(_vehs):
                    _veh_ids = [_veh.id for _veh in _vehs]
                    _veh_speeds = [_veh.speed for _veh in _vehs]

                    veh_ids_vels = zip(_veh_ids, _veh_speeds)
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
                num_delays = 0
                for time, phase_veh_ids in mem:
                    veh_ids = phase_veh_ids[p]
                    num_delays += len(veh_ids)
                tls_obs.append(num_delays)
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

