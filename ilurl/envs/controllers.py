"""Implementation of classic adaptive controllers and methods"""
import numpy as np

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
    def __init__(self, min_green, yellow, tls_ids, start_time_counter):
        self._tls_type = 'max_pressure'
        self._min_green = min_green
        self._yellow = yellow
        self._tls_phase = {tls_id: 0 for tls_id in tls_ids}
        self._tls_tc = {tls_id: start_time_counter for tls_id in tls_states}
        self._tls_ids = tls_ids
        self._tc = start_time_counter



    @property
    def tls_type(self):
        return self._tls_type

    def act(self, state):
        # 1) Update time for all tls_ids.
        self._tc += 1

        # 2) Get pressure values.
        pressure = state.feature_map(filter_by=('pressure',))

        # 3) Evaluate max pressure signal.
        # Filter for last change.
        ret = [False] * len(self._tls_ids)
        ind = 0
        for tls_id, lc in self._tls_tc.items():
            cp = self._tls_phase[ind]
            ret[ind], np = self._test_pressure(lc, cp, pressure[tls_id])

            # 4) if current p is different than last change
            # otherwise do nothing, extend.
            if ret[ind]:
                self._tls_tc[tls_id] = self._tc
                self._tls_phase[tls_id] = np
            else:
                ret[ind] = self._tc - lc == self._yellow:
            ind += 1
        return ret

    def _test_pressure(self, last_time_change, current_phase, pressures)
        if self._tc > self._min_green + last_time_change:
            next_phase = np.argmax(pressures)
            return next_phase != current_phase, next_phase
        return False, current_phase

    def reset(self):
        self._tls_phase = {tls_id: 0 for tls_id in tls_ids}

