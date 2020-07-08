import re
from collections import OrderedDict

import numpy as np


class Intersection:
    """Represents an intersection.

        * Nodes on a transportation network.
        * Root node on a tree hierarchy.
        * Basic unit on the feature map (an agent controls an intersection)
        * Splits outputs.
    """

    def __init__(self, mdp_params, tls_id, phases, phase_capacity):
        """Instanciate intersection object"""
        self._tls_id = tls_id

        # 2) Define children nodes: Phase
        self._phases = [Phase(mdp_params,
                              f'{tls_id}#{phase_id}',
                              phase_comp,
                              phase_capacity[phase_id])
                        for phase_id, phase_comp in phases.items()]


    @property
    def tls_id(self):
        return self._tls_id

    @property
    def phases(self):
        return self._phases

    def update(self, duration, vehs, tls):

        for p, phase in enumerate(self.phases):
            phase.update(duration, vehs[p], tls)

    def reset(self):
        for phase in self.phases:
            phase.reset()

    def feature_map(self, filter_by=None, categorize=True, split=False):
       ret = [phase.feature_map(filter_by, categorize)
              for phase in self.phases]

       if split:
           # num_features x num_phases 
           return tuple(zip(*ret))
       return ret


class Phase:
    """Represents a phase.

        * Composed of many lanes.
        * Performs categorization.
        * Aggregates wrt lanes.
        * Aggregates wrt time.
    """

    def __init__(self, mdp_params, phase_id, phase_data, max_capacity):
        # 1) Define base attributes
        self._phase_id = phase_id
        self._labels = mdp_params.states
        self._max_speed, self._max_count =  max_capacity
        self._matcher = re.compile('\[(.*?)\]')

        # 2) Get categorization bins.
        # fn: extracts category_<feature_name>s from mdp_params
        def fn(x):
            z = self._get_derived(x)
            return [getattr(mdp_params, y) for y in dir(mdp_params)
                    if (z in y) and ('category_' in y)][0]
        self._bins = {_feat: fn(_feat) for _feat in mdp_params.states}


        # 3) Instanciate lanes
        lanes = []
        components = []
        for _component in phase_data['components']:
            edge_id, lane_ids = _component
            for lane_id in lane_ids:
                components.append((edge_id, lane_id))
                lanes.append(Lane(mdp_params,
                                  f'{edge_id}#{lane_id}',
                                  self._max_speed))

        self._lanes = lanes
        self._components = components
        self._prev_features = {}

    @property
    def phase_id(self):
        return self._phase_id

    @property
    def labels(self):
        return self._labels

    @property
    def components(self):
        return self._components

    @property
    def lanes(self):
        return self._lanes

    def update(self, duration, vehs, tls):

        if duration == 0:
            # stores previous cycle for lag labels
            for label in self.labels:
                if 'lag' in label:
                    derived_label = self._get_derived(label)
                    self._prev_features[derived_label] = \
                                    getattr(self, derived_label)

        def _in(veh, lne):
            eid, lid = lne.lane_id.split('#')
            return veh.edge_id == eid and veh.lane == int(lid)

        for lane in self.lanes:
            _vehs = [v for v in vehs if _in(v, lane)]
            lane.update(duration, _vehs, tls)

    def reset(self):
        for lane in self.lanes:
            lane.reset()
        self._prev_features = {}

    def feature_map(self, filter_by=None, categorize=False):
        # 1) Select features.
        sel = [lbl for lbl in self.labels
           if filter_by is None or lbl in filter_by]

        # 2) Performs computes phases' features.
        ret = [self._get_feature_by(label) for label in sel]

        # 3) Categorize each phase feature.
        if categorize:
            ret = [np.digitize(val, bins=self._bins[self._get_derived(lbl)])
                   for val, lbl in zip(ret, sel)]

        return ret


    @property
    def speed(self):
        """Aggregates speed wrt time and lane

        Returns:
        -------
        * speed: float
            The average speed of all cars in the phase
        """
        K = len(self.lanes[0].cache)
        if K > 0:

            # TODO: Set no vehicles as nan
            total = 0
            prods = [0] * K
            counts = [0] * K
            for lane in self.lanes:
                # Sum of velocities / duration
                for i, s_c in enumerate(zip(lane.speeds, lane.counts)):
                    s, c  = s_c
                    prods[i] += s * c
                    counts[i] += c

            product = [p / c if c > 0.0 else 0.0 for p, c in zip(prods, counts)]
            return round(sum(product) / K, 2)
        # TODO: Return nan?
        return 0.0

    @property
    def count(self):
        """Aggregates count wrt time and lanes

        Returns:
        -------
        * count: float
            The average number of vehicles in the approach
        """
        K = len(self.lanes[0].cache)
        if K > 0:
            ret = sum([sum(lane.counts) for lane in self.lanes])
            return round(ret / K, 2)

        #TODO: Return nan?
        return 0

    @property
    def delay(self):
        """Aggregates delay wrt time and lanes

        Returns:
        -------
        * delay: float
            The average number of vehicles in delay in the cycle

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
        # TODO: Make average
        ret = 0
        for lane in self.lanes:
            ret += sum(lane.delays)

        # K = len(self.lanes[0].cache)
        return round(ret, 2)

    @property
    def queue(self):
        """Max. queue of vehicles wrt lane and time steps.

        Returns:
        -------
        * queue: int
            The maximum number of queued cars over all lanes

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
        # TODO: Divide by K?
        ret = 0
        for lane in self.lanes:
            ret = max(ret, max(lane.delays) if any(lane.delays) else 0)

        return round(ret, 2)

    def _get_feature_by(self, label):
        """Returns feature by label"""
        if 'lag' in label:
            derived_feature = \
                self._matcher.search(label).groups()[0]
            return self._prev_features[derived_feature]
        return getattr(self, label)

    def _get_derived(self, label):
        """Returns label or derived label"""
        derived_label = None
        if 'lag' in label:
            derived_label = \
                self._matcher.search(label).groups()[0]
        return derived_label or label


class Lane:
    """ Represents a lane within an edge.

        * Leaf nodes.
        * Caches observation data.
        * Performs normalization.
        * Computes feature per time step.
        * Aggregates wrt vehicles.
    """
    def __init__(self, mdp_params, lane_id, max_speed):
        self._lane_id = lane_id
        self._min_speed = mdp_params.velocity_threshold
        self._max_speed = max_speed
        self._normalize = mdp_params.normalize_state_space
        self._last_duration = 0
        self.reset()

    @property
    def lane_id(self):
        return self._lane_id

    @property
    def cache(self):
        return self._cache

    def reset(self):
        self._cache = OrderedDict()
        self._last_duration = 0

    def update(self, duration, vehs, tls):
        self._cache[duration] = (vehs, tls)
        self._last_duration = duration

    @property
    def speeds(self):
        """Vehicles' speeds per time step.

        Returns:
            speeds: list<float>
            Is a duration sized list containing averages
        """
        cap = self._max_speed if self._normalize else 1

        # 1) IF no cars are available revert to zero
        ret = [np.nanmean([v.speed for v in vehs]) / cap if any(vehs) else 0.0
                for vehs, _ in self.cache.values()]

        return ret

    @property
    def counts(self):
        """Number of vehicles per time step.

        Returns:
            count: list<int>
            Is a duration sized list containing the total number of vehicles
        """
        ret = [len(vehs) for vehs, _ in self.cache.values()]
        return ret

    @property
    def delays(self):
        """Total of vehicles circulating under a velocity threshold

        Returns:
        -------
            * delay: list<int>
            Is a duration sized list containing the total number of slow moving
            vehicles.
        """
        cap = self._max_speed if self._normalize else 1
        ds = self._min_speed

        vehs = [vehs_tls[0] if any(vehs_tls[0]) else []
                for d, vehs_tls in self.cache.items() if d <= self._last_duration]

        ret = [sum([v.speed / cap < ds for v in veh]) if any(veh) else 0
                for veh in vehs]
        return ret

