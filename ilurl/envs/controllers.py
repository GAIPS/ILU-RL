"""Implementation of classic adaptive controllers and methods"""
import numpy as np

from collections import namedtuple

PressurePhase = namedtuple('PressurePhase', 'id time yellow')

def is_controller_periodic(ts_type):
    if ts_type in ('rl', 'static', 'uniform', 'random'):
        return True

    if ts_type in ('actuated','actuated_delay', 'max_pressure'):
        return False

    raise ValueError(f'Unknown ts_type:{ts_type}')

def get_ts_controller(ts_type, ts_num_phases):
    if ts_type in ('max_pressure',):
       return MaxPressure(12, 120, 6, ts_num_phases)

    raise ValueError('Only max-pressure controller implemented')

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
        self._ts_phase = {ts_id: 0 for ts_id in ts_ids}

