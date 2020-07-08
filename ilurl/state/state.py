import numpy as np

from ilurl.state.elements import Intersection
from ilurl.utils.aux import flatten as flat


class State:
    """Describes the state space

        * Forest: composed of root nodes of trees (intersections).
        * Receives kernel data from vehicles (e.g speed) and intersections (e.g states).
        * Broadcasts data to intersections.
        * Outputs features.
        * Computes global features e.g Time
    """

    def __init__(self, network, mdp_params):
        """
        Params:
        ------
        * network: ilurl.networks.base.Network
            network to be described by states

        * mdp_params: ilurl.core.params.MDPParams
            mdp specification: agent, states, rewards, gamma and learning params

        Returns:
        --------

        * state
        """
        self._tls_ids = network.tls_ids
        # Global features: e.g Time
        self._time = -1
        self._has_time = False
        #if 'time' in mdp_params.states and mdp_params.time_period is not None:
        if mdp_params.time_period is not None:
            self._has_time = True
            self._bins = mdp_params.category_times
            self._last_time = -1
            self._time_period = mdp_params.time_period


        # Local features.
        self._intersections = []

        # Local feature
        self._intersections = {
            tls_id: Intersection(mdp_params,
                                 tls_id,
                                 network.tls_phases[tls_id],
                                 network.tls_max_capacity[tls_id])
            for tls_id in network.tls_ids}

    @property
    def tls_ids(self):
        return self._tls_ids

    @property
    def time(self):
        return self._time

    def update(self, duration, vehs, tls=None):
        # 1) Update time.
        self._update_time(duration)

        # 2) Broadcast update to intersections.
        for tls_id, i in self._intersections.items():
            i.update(duration, vehs[tls_id], tls)

    def reset(self):
        self._last_time = -1
        self._time = -1
        for intersection in self._intersections.values():
            intersection.reset()

    def feature_map(self, filter_by=None, categorize=False, split=False, flatten=False):
        # 1) Delegate feature computation to tree.
        ret = {k:v.feature_map(filter_by=filter_by,
                               categorize=categorize,
                               split=split)
               for k, v in self._intersections.items()}


        # 2) Add network features.
        if self._has_time:
            ret = {k:self._add_time(filter_by, categorize, split, v)
                   for k, v in ret.items()}

        # 3) Flatten results.
        if flatten:
            ret = {k: tuple(flat(v)) for k, v in ret.items()}
        return ret

    def _update_time(self, duration):
        """Update time as feature"""
        if self._has_time and self._last_time != duration:
            self._time += 1
            self._last_time = duration

    def _add_time(self, filter_by, categorize, split, features):
        """Add time as feature to intersections"""
        if self._has_time:
            period = self._time_period
            # 1) Verify time conditions.
            if filter_by is None or ('time' in filter_by):
                # 2) Number of periods (e.g hours) % Rolling update.
                ret  = (self.time // period) % int(24 * 3600 / period)

                # 3) Convert into category.
                if categorize:
                    ret = np.digitize(ret, self._bins)

                # 4) Add to features.
                if split:
                    ret = tuple([ret] + list(features))
                else:
                    ret = [ret] + list(features)
                return ret
            return features
        return features
