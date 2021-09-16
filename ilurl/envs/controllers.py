"""Implementation of classic adaptive controllers and methods"""
import copy

import numpy as np

from collections import namedtuple

PressurePhase = namedtuple('PressurePhase', 'id time yellow')

def is_controller_periodic(ts_type):
    if ts_type in ( 'static', 'webster'):
        return True

    if ts_type in ('actuated', 'max_pressure', 'rl', 'centralized', 'cg', 'random'):
        return False

    raise ValueError(f'Unknown ts_type:{ts_type}')

def get_ts_controller(ts_type, ts_num_phases, tls_phases, cycle_time):
    if ts_type == 'max_pressure':
       return MaxPressure(12, 120, 6, ts_num_phases)
    elif ts_type == 'webster':
        return Webster(300, tls_phases, cycle_time)
    raise ValueError(f'Unknown ts_type:{ts_type}')

class MaxPressure:
    """Adaptive rule based controller based of OSFATSC

        Reference:
        ----------
        * Wade Genders and Saiedeh Razavi, 2019
            An Open Source Framework for Adaptive Traffic Signal Control.

        See also:
        ---------
        * Wei, et al, 2018
            PressLight: Learning Max Pressure Control to Coordinate Traffic
                        Signals in Arterial Network.

        * Pravin Varaiya, 2013
            Max pressure control of a network of signalized intersections

    """
    def __init__(self, min_green, max_green, yellow, ts_num_phases):
        # Validate input arguments
        # TODO: Extend the case for larger than 2
        assert all(nump == 2 for nump in ts_num_phases.values()), 'number of phases must equal 2'
        assert min_green > yellow, 'yellow must be lesser than min_green'
        assert max_green > min_green, 'min_green must be lesser than max_green'
        
        # controller configuration parameters
        self._ts_type = 'max_pressure'
        self._min_green = min_green
        self._max_green = max_green
        self._yellow = yellow

        # controller + network parameters
        # current phase and current timer
        ts_ids = list(ts_num_phases.keys())
        self._ts_phase = {ts_id: PressurePhase(0, 0, -1) for ts_id in ts_ids}

    @property
    def ts_type(self):
        return self._ts_type

    def act(self, state, tc):
        # 1) Get pressure values.
        pressure = state.feature_map(filter_by=('pressure',),
                                     split=True,
                                     flatten=True)

        # 2) Evaluate max pressure signal.
        # Filter for last change.
        ret = [False] * len(self._ts_phase)
        ind = 0

        for ts_id, pp in self._ts_phase.items():
            press = pressure[ts_id]
            ret[ind], *data = self._switch_pressure(press, pp, tc)
            self._ts_phase[ts_id] = PressurePhase(*data)
            ind += 1

        return ret

    def _switch_pressure(self, pressure, pressure_phase, tc):
        pid, pt, py = pressure_phase
        # 1) Yellow expired.
        if tc == py:
            return True, pid, pt, tc

        # 1) Hard change: too much time has gone by without change
        if tc > self._max_green + self._yellow + pt:
            # Circular update to the next phase 
            return True, (pid + 1) % 2, tc, tc + self._yellow

        # 2) Adaptive change: evaluate pressure
        if tc > self._min_green + self._yellow + pt:
            next_phase = np.argmax(pressure)
            switch = next_phase != pid
            pt = switch * tc + (not switch) * pt
            py = switch * (pt + self._yellow) + (not switch) * py
            return switch, next_phase, pt, py

        # 3) Do nothing
        return False, pid, pt, py

    def reset(self):
        self._ts_phase = {ts_id: PressurePhase(0, 0, -1) for ts_id in self._ts_phase}

    def terminate(self):
        self.reset()

class Webster:
    """
        Adaptive webster method.
    """
    def __init__(self, aggregation_period, tls_phases, cycle_time):

        self._ts_type = 'webster'

        # Internalise parameters.
        self._aggregation_period = aggregation_period
        self._cycle_time = cycle_time
        self._tls_phases = tls_phases

        # Initialise vehicles counts data structure.
        self._vehicles_counts = {nid: {p: {e[0]: {l: [] for l in e[1]}
                                for e in data['incoming']}
                                    for p, data in self._tls_phases[nid].items()}
                                        for nid in self._tls_phases}

        # Uncomment below for (static) Webster timings calculation.
        self._global_counts = {nid: {p: {e[0]: {l: [] for l in e[1]}
                                for e in data['incoming']}
                                    for p, data in self._tls_phases[nid].items()}
                                        for nid in self._tls_phases}

        # Calculate uniform timings.
        self._uniform_timings = {}
        for tid in self._tls_phases:
            timings = []

            num_phases = len(self._tls_phases[tid])

            # Calculate ratios.
            ratios = [1/num_phases for p in range(num_phases)]

            # Calculate phases durations given allocation ratios.
            phases_durations = [np.around(r*(cycle_time-6.0*num_phases)) for r in ratios]

            # Calculate timings.
            counter = 0
            timings = []
            for p in range(num_phases):
                timings.append(counter + phases_durations[p])
                timings.append(counter + phases_durations[p] + 6.0)
                counter += phases_durations[p] + 6.0

            timings[-1] = self._cycle_time
            timings[-2] = self._cycle_time - 6.0

            self._uniform_timings[tid] = timings

        self._webster_timings = copy.deepcopy(self._uniform_timings) # Initialize webster with uniform timings.
        self._next_signal_plan = copy.deepcopy(self._uniform_timings)

        # Internal counter.
        self._time_counter = 1

    @property
    def ts_type(self):
        return self._ts_type

    def act(self, kernel_data):

        # Update counts.
        for nid in kernel_data:
            for p, vehs_p in kernel_data[nid].items():
                for veh in vehs_p:
                    if veh.id not in self._vehicles_counts[nid][p][veh.edge_id][veh.lane]:
                        self._vehicles_counts[nid][p][veh.edge_id][veh.lane].append(veh.id)

                    # Uncomment below for (static) Webster timings calculation.
                    if veh.id not in self._global_counts[nid][p][veh.edge_id][veh.lane]:
                        self._global_counts[nid][p][veh.edge_id][veh.lane].append(veh.id)

        if (self._time_counter % self._aggregation_period == 0) and self._time_counter > 1:
            # Calculate new signal plan.

            for tls_id in self._vehicles_counts.keys():
                max_counts = []
                for p in self._vehicles_counts[tls_id].keys():
                    max_count = -1
                    for edge in self._vehicles_counts[tls_id][p].keys():
                        for l in self._vehicles_counts[tls_id][p][edge].keys():
                            lane_count = len(self._vehicles_counts[tls_id][p][edge][l])
                            max_count = max(max_count, lane_count)
                    max_counts.append(max_count)

                num_phases = len(max_counts)

                if min(max_counts) < 2:

                    # Use global counts to calculate timings.
                    max_counts = []
                    for p in self._global_counts[tls_id].keys():
                        max_count = -1
                        for edge in self._global_counts[tls_id][p].keys():
                            for l in self._global_counts[tls_id][p][edge].keys():
                                lane_count = len(self._global_counts[tls_id][p][edge][l])
                                max_count = max(max_count, lane_count)
                        max_counts.append(max_count)

                    # Calculate ratios.
                    ratios = [p/sum(max_counts) for p in max_counts]

                    # Calculate phases durations given allocation ratios.
                    phases_durations = [np.around(r*(self._cycle_time-6.0*num_phases)) for r in ratios]

                    # Calculate timings.
                    counter = 0
                    timings = []
                    for p in range(num_phases):
                        timings.append(counter + phases_durations[p])
                        timings.append(counter + phases_durations[p] + 6.0)
                        counter += phases_durations[p] + 6.0

                    timings[-1] = self._cycle_time
                    timings[-2] = self._cycle_time - 6.0

                    self._next_signal_plan[tls_id] = timings

                else:
                    # Use counts from the aggregation period to calculate timings.

                    # Calculate ratios.
                    ratios = [p/sum(max_counts) for p in max_counts]

                    # Calculate phases durations given allocation ratios.
                    phases_durations = [np.around(r*(self._cycle_time-6.0*num_phases)) for r in ratios]

                    # Calculate timings.
                    counter = 0
                    timings = []
                    for p in range(num_phases):
                        timings.append(counter + phases_durations[p])
                        timings.append(counter + phases_durations[p] + 6.0)
                        counter += phases_durations[p] + 6.0

                    timings[-1] = self._cycle_time
                    timings[-2] = self._cycle_time - 6.0

                    self._next_signal_plan[tls_id] = timings

            # Reset counters.
            self._reset_counts()

        if (self._time_counter % self._cycle_time == 0) and self._time_counter > 1:
            # Update current signal plan.
            self._webster_timings = copy.deepcopy(self._next_signal_plan)

        # Increment internal counter.
        self._time_counter += 1

        return self._webster_timings

    def _reset_counts(self):
        self._vehicles_counts = {nid: {p: {e[0]: {l: [] for l in e[1]}
                        for e in data['incoming']}
                            for p, data in self._tls_phases[nid].items()}
                                for nid in self._tls_phases}

    def terminate(self):
        self._reset_counts()

        # Uncomment below for (static) Webster timings calculation.
        """ global_timings = {}

        for tls_id in self._global_counts.keys():
            max_counts = []
            for p in self._global_counts[tls_id].keys():
                max_count = -1
                for edge in self._global_counts[tls_id][p].keys():
                    for l in self._global_counts[tls_id][p][edge].keys():
                        lane_count = len(self._global_counts[tls_id][p][edge][l])
                        max_count = max(max_count, lane_count)
                max_counts.append(max_count)

            num_phases = len(max_counts)

            # Calculate ratios.
            ratios = [p/sum(max_counts) for p in max_counts]

            # Calculate phases durations given allocation ratios.
            phases_durations = [np.around(r*(self._cycle_time-6.0*num_phases)) for r in ratios]

            # Calculate timings.
            counter = 0
            timings = []
            for p in range(num_phases):
                timings.append(counter + phases_durations[p])
                timings.append(counter + phases_durations[p] + 6.0)
                counter += phases_durations[p] + 6.0

            timings[-1] = self._cycle_time
            timings[-2] = self._cycle_time - 6.0

            global_timings[tls_id] = timings

        print('Global timings (Webster method):', global_timings) """
