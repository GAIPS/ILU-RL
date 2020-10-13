import numpy as np

from ilurl.utils.aux import flatten as flat
from ilurl.state.node import Node
from ilurl.state.intersection import Intersection

class State(Node):
    """Describes the state space

        * Forest: composed of root nodes of trees (intersections).
        * Receives kernel data from vehicles (e.g speed) and
          intersections (e.g states).
        * Broadcasts data to intersections.
        * Outputs features.
        * Computes global features e.g Time
    """

    def __init__(self, network, mdp_params):
        """Instantiate State object

            * Singleton object that produces features for all the network.

        Params:
        ------
        * network: ilurl.networks.base.Network
            network to be described by states

        * mdp_params: ilurl.core.params.MDPParams
            mdp specification: agent, states, rewards, gamma and learning params

        Returns:
        --------
        * state
            A state is an aggregate of features indexed by intersections' ids.
        """
        self._tls_ids = network.tls_ids

        # Global features: e.g Time
        self._time = -1
        self._has_time = False
        if mdp_params.time_period is not None:
            self._has_time = True
            self._bins = mdp_params.category_times
            self._last_time = -1
            self._time_period = mdp_params.time_period

        self._labels = mdp_params.features
        # Local features.
        intersections = {
            tls_id: Intersection(self,
                                 mdp_params,
                                 tls_id,
                                 network.tls_phases[tls_id],
                                 network.tls_max_capacity[tls_id])
            for tls_id in network.tls_ids}

        super(State, self).__init__(None, 'state', intersections)

    @property
    def tls_ids(self):
        return self._tls_ids

    @property
    def labels(self):
        return self._labels

    def update(self, duration, vehs, tls):
        """Update data structures with observation space

            * Broadcast it to intersections

        Params:
        -------
            * duration: float
                Number of time steps in seconds within a cycle.
                Assumption: min time_step 1 second.
                Circular updates: 0.0, 1.0, 2.0, ..., 0.0, 1.0, ...

            * vehs: list<namedtuple<ilurl.envs.elements.Vehicles>>
                Container for vehicle data (from VehicleKernel)

            * tls: list<namedtuple<ilurl.envs.elements.TrafficLight>>
                Container for traffic light program representation
        """
        # 1) Update time.
        self._update_time(duration)

        # 2) Broadcast update to intersections.
        for tl_id, inter in self.intersections.items():
            inter.update(duration, vehs[tl_id], tls[tl_id])

        # 3) Additional intersection commands
        for inter in self.intersections.values():
            inter.after_update()


    def reset(self):
        """Clears data from previous cycles, broadcasts method to phases"""
        self._last_time = -1
        self._time = -1
        for intersection in self.intersections.values():
            intersection.reset()

    def feature_map(self, filter_by=None, categorize=False,
                    split=False, flatten=False):
        """Computes features for all intersections

            * Computes network wide features. (e.g time)
            * Delegates for each intersection it's features construction.

        Params:
        -------
            * filter_by: list<str>
                select the labels of the features to output.

            * categorize: bool
                discretizes the output according to preset categories.

            * split: bool
                groups outputs by features, not by phases.

            * flatten: bool
                the results are not nested wrt to phases or features.

        Returns:
        -------
            * intersection_features: dict<str, list<list<numeric>>>
                Keys are junction ids
                Each nested list represents the phases's features.
                If categorize then numeric is integer (category) else is float.
                If split then groups features instead of phases.
                If flatten then nested list becomes flattened.
        """
        # 1) Validate user input
        if filter_by is not None:
            if not (set(filter_by).issubset(set(self.labels))):
                err = f'filter_by {filter_by} must belong to states {self.labels}'
                raise ValueError(err)

        # 2) Delegate feature computation to tree.
        ret = {k:v.feature_map(filter_by=filter_by,
                               categorize=categorize,
                               split=split)
               for k, v in self.intersections.items()}


        # 3) Add network features.
        if self._has_time:
            ret = {k:self._add_time(filter_by, categorize, split, v)
                   for k, v in ret.items()}

        # 4) Flatten results.
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
        period = self._time_period
        # 1) Verify time conditions.
        if filter_by is None or ('time' in filter_by):
            # 2) Number of periods (e.g hours) % Rolling update.
            ret  = (self._time // period) % int(24 * 3600 / period)

            # 3) Convert into category.
            if categorize:
                ret = int(np.digitize(ret, self._bins))

            # 4) Add to features.
            if split:
                ret = tuple([ret] + list(features))
            else:
                ret = [ret] + list(features)
            return ret
        return features

