"""Implementation of classic adaptive controllers and methods"""
import numpy as np

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
        self._ts_cptp = {ts_id: (0, 0) for ts_id in ts_ids}
        self._ts_ids = ts_ids


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
        ret = [False] * len(self._ts_ids)
        ind = 0

        for ts_id, cptp in self._ts_cptp.items():
            cp, tp = cptp
            press = pressure[ts_id]
            ret[ind], next_phase = self._test_pressure(cp, tp, tc, press)

            # 3) if current p is different than last change
            # otherwise do nothing, extend.
            if ret[ind]:
                self._ts_cptp[ts_id] = (next_phase, tc)
            else:
                ret[ind] = (tc - tp) == self._yellow
            ind += 1
        return ret

    def _test_pressure(self, current_phase, last_time_change, time_counter, press_phase):
        # 1) Hard change: too much time has gone by without change
        if time_counter > self._max_green + last_time_change:
            # Circular update to the next phase 
            return True, (current_phase + 1) % 2

        # 2) Adaptive change: evaluate pressure
        if time_counter > self._min_green + last_time_change:
            next_phase = np.argmax(press_phase)
            return next_phase != current_phase, next_phase

        # 3) Do nothing
        return False, current_phase

    def reset(self):
        self._ts_phase = {ts_id: 0 for ts_id in ts_ids}

