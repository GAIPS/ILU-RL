import numpy as np

class Intersection:
    """Represents an intersection.

        * Nodes on a transportation network
        * Root node on a tree hierarchy
        * Basic unit on the feature map (an agent controls an intersection)
        * Splits outputs
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
        self._phase_id = phase_id

        # 1) Get categorization bins
        # fn: extracts category_<feature_name>s from mdp_params
        def fn(x):
            return [getattr(mdp_params, y) for y in dir(mdp_params)
                    if (x in y) and ('category_' in y)][0]
        self._bins = {_feat: fn(_feat) for _feat in mdp_params.states}

        # TODO: Rename states to features
        self._labels = mdp_params.states
        self._normalize = mdp_params.normalize_state_space
        self._max_speed, self._max_count =  max_capacity
        self._velocity_threshold = mdp_params.velocity_threshold

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

        def _in(veh, lne):
            eid, lid = lne.lane_id.split('#')
            return veh.edge_id == eid and veh.lane == int(lid)

        for lane in self.lanes:
            _vehs = [v for v in vehs if _in(v, lane)]
            lane.update(duration, _vehs, tls)

    def feature_map(self, filter_by=None, categorize=True):
        # 1) Select features.
        sel = [lbl for lbl in self.labels
           if filter_by is None or lbl in filter_by]

        # 2) Performs computes phases' features.
        ret = [getattr(self, label) for label in sel]

        # 3) Categorize each phase feature.
        if categorize:
            ret = [np.digitize(val, bins=self._bins[lbl])
                   for val, lbl in zip(ret, sel)]
        return ret


    @property
    def speed(self):
        """Aggregates and normalizes speed wrt time and lane"""
        cap = self._max_speed if self._normalize else 1

        # TODO: Set no vehicles as nan
        total = 0
        K = len(self.lanes[0].cache)
        prods = [0] * K
        counts = [0] * K
        for lane in self.lanes:
            # Sum of velocities / duration
            for i, s_c in enumerate(zip(lane.speed, lane.count)):
                s, c  = s_c
                prods[i] += s * c
                counts[i] += c

        product = [p / c if c > 0.0 else 0.0 for p, c in zip(prods, counts)]
        return round(sum(product) / (cap * K), 2)

    @property
    def count(self):
        """Aggregates count wrt time and lanes"""
        # disable for now
        # cap = self._max_count if self._normalize else 1
        ret = sum([sum(lane.count) for lane in self.lanes])
        K = len(self.lanes[0].cache)
        return round(ret / K, 2)

    @property
    def delay(self):
        """Aggregates delay wrt time and lanes"""
        # TODO: Make average
        ret = 0
        for lane in self.lanes:
            ret += sum(lane.delay)

        # K = len(self.lanes[0].cache)
        return round(ret, 2)

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
        self._cache = {}

    def update(self, duration, vehs, tls):
        self._cache[duration] = (vehs, tls)
        self._last_duration = duration

    @property
    def speed(self):
        """Mean speed per lane and time step"""
        # 1) IF no cars are available revert to zero
        ret = [np.nanmean([v.speed for v in vehs]) if any(vehs) else 0.0
                for vehs, _ in self.cache.values()]

        return ret

    @property
    def count(self):
        """Number of vehicles per lane and time step"""
        ret = [len(vehs) for vehs, _ in self.cache.values()]
        return ret

    @property
    def delay(self):
        """Every cars' speed per lane and time step"""
        cap = self._max_speed if self._normalize else 1
        ds = self._min_speed

        vehs = [vehs_tls[0] if any(vehs_tls[0]) else []
                for d, vehs_tls in self.cache.items() if d <= self._last_duration]

        ret = [sum([v.speed / cap < ds for v in veh]) if any(veh) else 0
                for veh in vehs]
        return ret

