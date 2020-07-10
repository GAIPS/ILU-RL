import re
from collections import OrderedDict

import numpy as np

from ilurl.utils.aux import flatten as flat


class State:
    """Describes the state space

        * Forest: composed of root nodes of trees (intersections).
        * Receives kernel data from vehicles (e.g speed) and
          intersections (e.g states).
        * Broadcasts data to intersections.
        * Outputs features.
        * Computes global features e.g Time
    """

    def __init__(self, network, mdp_params):
        """Instanciate State object

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

    def update(self, duration, vehs, tls=None):
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

            * tls: list<namedtuple<ilurl.envs.elements.TrafficLightSignal>>
                Container for traffic light program representation
        """
        # 1) Update time.
        self._update_time(duration)

        # 2) Broadcast update to intersections.
        for tls_id, i in self._intersections.items():
            i.update(duration, vehs[tls_id], tls)

    def reset(self):
        """Clears data from previous cycles, broadcasts method to phases"""
        self._last_time = -1
        self._time = -1
        for intersection in self._intersections.values():
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
        return features

class Intersection:
    """Represents an intersection.

        * Nodes on a transportation network.
        * Root node on a tree hierarchy.
        * Basic unit on the feature map (an agent controls an intersection)
        * Splits outputs.
    """

    def __init__(self, mdp_params, tls_id, phases, phase_capacity):
        """Instanciate intersection object

        Params:
        ------
        * network: ilurl.networks.base.Network
            network to be described by states

        * mdp_params: ilurl.core.params.MDPParams
            mdp specification: agent, states, rewards, gamma and learning params

        * tls_id: str
            an id for the traffic light signal

        * phases: dict<int, (str, list<int>)>
            key is the index of the phase.
            tuple is the phase component where:
                str is the edge_id
                list<int> is the list of lane_ids which belong to phase.

        * phase_capacity: dict<int, (float, float)>
            key is the index of the phase.
            tuple is the maximum capacity where:
                float is the max speed of vehicles for the phase.
                float is the max count of vehicles for the phase.

        Returns:
        --------
        * state
            A state is an aggregate of features indexed by intersections' ids.
        """
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
        """Update data structures with observation space

            * Broadcast it to phases

        Params:
        -------
            * duration: float
                Number of time steps in seconds within a cycle.
                Assumption: min time_step 1 second.
                Circular updates: 0.0, 1.0, 2.0, ..., 0.0, 1.0, ...

            * vehs: list<namedtuple<ilurl.envs.elements.Vehicles>>
                Container for vehicle data (from VehicleKernel)

            * tls: list<namedtuple<ilurl.envs.elements.TrafficLightSignal>>
                Container for traffic light program representation
        """
        for p, phase in enumerate(self.phases):
            phase.update(duration, vehs[p], tls)

    def reset(self):
        """Clears data from previous cycles, broadcasts method to phases"""
        for phase in self.phases:
            phase.reset()

    def feature_map(self, filter_by=None, categorize=True, split=False):
        """Computes intersection's features

        Params:
        -------
            * filter_by: list<str>
                select the labels of the features to output.

            * categorize: bool
                discretizes the output according to preset categories.

            * split: bool
                groups outputs by features, not by phases.

        Returns:
        -------
            * phase_features: list<list<float>> or list<list<int>>
                Each nested list represents the phases's features.
        """
        ret = [phase.feature_map(filter_by, categorize)
              for phase in self.phases]

        if split:
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
        """Builds phase

        Params:
        ------
        * mdp_params: ilurl.core.params.MDPParams
            mdp specification: agent, states, rewards, gamma and
                               learning params

        * phase_id: int
            key is the index of the phase.

        * phase_data: (str, list<int>)
            tuple is the phase component where:
                str is the edge_id
                list<int> is the list of lane_ids which belong to phase.

        * max_capacity: (float, float)
            float is the max count of vehicles for the phase.
            float is the max speed of vehicles for the phase.

        """
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
                lanes.append(
                    Lane(mdp_params, edge_id, lane_id, self._max_speed))

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
        """Update data structures with observation space

            * Updates also bound lanes.

        Params:
        -------
            * duration: float
                Number of time steps in seconds within a cycle.
                Assumption: min time_step 1 second.
                Circular updates: 0.0, 1.0, 2.0, ..., 0.0, 1.0, ...

            * vehs: list<namedtuple<ilurl.envs.elements.Vehicles>>
                Container for vehicle data (from VehicleKernel)

            * tls: list<namedtuple<ilurl.envs.elements.TrafficLightSignal>>
                Container for traffic light program representation
        """
        # 1) Ignores update for the same duration.
        if duration != self._last_update:
            if duration == 0:
                # 2) Stores previous cycle for lag labels.
                for label in self.labels:
                    if 'lag' in label:
                        derived_label = self._get_derived(label)
                        self._prev_features[derived_label] = \
                                        getattr(self, derived_label)

                # 3) Reset data for new cycle.
                # TODO: remove this after refactor (redundant)
                self.reset(reset_previous_features=False)

            # 4) Define a help function filtering function.
            def _in(veh, lane):
                return veh.edge_id == lane.edge_id and veh.lane == lane.lane_id

            step_speed = 0
            step_count = 0
            step_delay = 0
            for lane in self.lanes:
                _vehs = [v for v in vehs if _in(v, lane)]
                lane.update(duration, _vehs, tls)


                step_count += lane._cached_counts2
                step_speed += lane._cached_speeds2
                step_delay += lane._cached_delays2

            self._update_counter += 1
            self._last_update = duration

            K = self._update_counter
            speed_update = (K - 1) * self._cached_speed
            if step_count > 0:
                speed_update += step_speed / step_count
            self._cached_speed = speed_update / K


            count_update = (K - 1) * self._cached_count
            self._cached_count  = (step_count + count_update) / K

            self._cached_delay  += step_delay


    def reset(self, reset_previous_features=True):
        """Clears data from previous cycles, broadcasts method to lanes"""
        for lane in self.lanes:
            lane.reset()

        if reset_previous_features:
            self._prev_features = {}

        # Test variables
        self._cached_speed = 0
        self._cached_count = 0
        self._cached_delay = 0

        self._update_counter = 0
        self._last_update = -1

    def feature_map(self, filter_by=None, categorize=False):
        """Computes phases' features

        Params:
        -------
            * filter_by: list<str>
                select the labels of the features to output.

            * categorize: bool
                discretizes the output according to preset categories.

        Returns:
        -------
            * phase_features: list<float> or list<int>
                len(.) == number of selected labels.
                if categorized is set to true then list<int>
        """
        # 1) Select features.
        sel = [lbl for lbl in self.labels
           if filter_by is None or lbl in filter_by]

        # 2) Performs computes phases' features.
        ret = [self._get_feature_by(label) for label in sel]

        # 3) Categorize each phase feature.
        if categorize:
            ret = [self._digitize(val, lbl) for val, lbl in zip(ret, sel)]

        return ret


    @property
    def speed(self):
        """Aggregates speed wrt time and lane

        Returns:
        -------
        * speed: float
            The average speed of all cars in the phase
        """
        # K = len(self.lanes[0].cache)
        # ret = 0.0
        # if K > 0:
        #     # TODO: Set no vehicles as nan
        #     total = 0
        #     prods = [0] * K
        #     counts = [0] * K
        #     for lane in self.lanes:
        #         # Sum of velocities / duration
        #         for i, s_c in enumerate(zip(lane.speeds, lane.counts)):
        #             s, c  = s_c
        #             prods[i] += s * c
        #             counts[i] += c

        #     product = [p / c if c > 0.0 else 0.0 for p, c in zip(prods, counts)]
        #     ret = round(sum(product) / K, 2)

        #     # test new process
        #     try:
        #         assert round(self._cached_speed, 2) == ret
        #     except AssertionError:
        #         import ipdb
        #         ipdb.set_trace()

        return round(self._cached_speed, 2)
        # TODO: Return nan?
        # return ret

    @property
    def count(self):
        """Aggregates count wrt time and lanes

        Returns:
        -------
        * count: float
            The average number of vehicles in the approach
        """
        # K = len(self.lanes[0].cache)
        # if K > 0:
        #     ret = sum([sum(lane.counts) for lane in self.lanes])
        #     return round(ret / K, 2)

        #     # test new process
        #     try:
        #         assert round(self._cached_count, 2) == ret
        #     except AssertionError:
        #         import ipdb
        #         ipdb.set_trace()

        # return ret

        # #TODO: Return nan?
        # return 0
        return round(self._cached_count, 2)

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
            "Design for Reinforcement Learning Params for Seamless"

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
        # # TODO: Make average
        # ret = 0
        # for lane in self.lanes:
        #     ret += sum(lane.delays)

        # try:
        #     assert round(self._cached_delay, 2) == round(ret, 2)
        # except AssertionError:
        #     import ipdb
        #     ipdb.set_trace()

        # return round(ret, 2)
        return round(self._cached_delay, 2)

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
            "Design for Reinforcement Learning Params for Seamless"

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

    def _digitize(self, value, label):
        _bins = self._bins[self._get_derived(label)]
        return int(np.digitize(value, bins=_bins))

class Lane:
    """ Represents a lane within an edge.

        * Leaf nodes.
        * Caches observation data.
        * Performs normalization.
        * Computes feature per time step.
        * Aggregates wrt vehicles.
    """
    def __init__(self, mdp_params, edge_id, lane_id, max_speed):
        """Builds lane

        Params:
        ------
        * mdp_params: ilurl.core.params.MDPParams
            mdp specification: agent, states, rewards, gamma and
                               learning params

        * lane_id: int
            key is the index of the lane.

        * max_speed: float
            max velocity a car can travel.
        """
        self._edge_id = edge_id
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
    def edge_id(self):
        return self._edge_id

    @property
    def cache(self):
        return self._cache

    def reset(self):
        """Clears data from previous cycles, define data structures"""
        self._cache = OrderedDict()
        self._cached_speeds = []
        self._cached_counts = []
        self._cached_delays = []


        # Test variables
        self._cached_speeds2 = (-1, 0)
        self._cached_counts2 = (-1, 0)
        self._cached_delay2 = (-1, 0)
        self._last_duration = 0


    def update(self, duration, vehs, tls):
        """Update data structures with observation space

            * Holds vehs and tls data for the duration of a cycle

            * Updates features of interest for a time step.

        Params:
        -------
            * duration: float
                Number of time steps in seconds within a cycle.
                Assumption: min time_step 1 second.
                Circular updates: 0.0, 1.0, 2.0, ..., 0.0, 1.0, ...

            * vehs: list<namedtuple<ilurl.envs.elements.Vehicles>>
                Container for vehicle data (from VehicleKernel)

            * tls: list<namedtuple<ilurl.envs.elements.TrafficLightSignal>>
                Container for traffic light program representation
        """
        # TODO: As of now stores an array with the cycles' data
        # More efficient to only store last time_step
        # cross sectional data or intra time_step data.
        self._cache[duration] = (vehs, tls)
        self._last_duration = duration

        self._update_counts()
        self._update_delays()
        self._update_speeds()

    def _update_speeds(self):
        """Step update for speeds variable"""
        # 1) Retrieve data 
        cap = self._max_speed if self._normalize else 1
        vehs, _ = self.cache[self._last_duration]

        # 2) Compute speed @ duration time step
        step_speed = \
            np.nanmean([v.speed for v in vehs]) / cap if any(vehs) else 0.0

        # # 3) Append or update
        # if self._last_duration == len(self._cached_speeds):
        #     self._cached_speeds.append(step_speed)
        # else:
        #     self._cached_speeds[int(self._last_duration)] = step_speed

        # 4) new state
        self._cached_speeds2 = \
            sum([v.speed for v in vehs]) / cap if any(vehs) else 0.0


    def _update_counts(self):
        """Step update for counts variable"""
        # 1) Retrieve data 
        vehs, _ = self.cache[self._last_duration]

        # 2) Compute count @ duration time step
        step_count = len(vehs)

        # # 3) Append or update
        # if self._last_duration == len(self._cached_counts):
        #     self._cached_counts.append(step_count)
        # else:
        #     self._cached_counts[int(self._last_duration)] = step_count

        # 4) new state
        self._cached_counts2 = step_count

    def _update_delays(self):
        """Step update for delays variable"""
        # 1) Retrieve data 
        cap = self._max_speed if self._normalize else 1
        vt = self._min_speed
        vehs, _ = self.cache[self._last_duration]

        # 2) Compute speed @ duration time step
        step_delay = len([v.speed / cap < vt for v in vehs])

        # 3) Append or update
        # if self._last_duration == len(self._cached_delays):
        #     self._cached_delays.append(step_delay)
        # else:
        #     self._cached_delays[int(self._last_duration)] = step_delay

        # 4) new state
        self._cached_delays2 = step_delay

    @property
    def speeds(self):
        """Vehicles' speeds per time step.

        Returns:
            speeds: list<float>
            Is a duration sized list containing averages
        """
        return self._cached_speeds2

    @property
    def counts(self):
        """Number of vehicles per time step.

        Returns:
            count: list<int>
            Is a duration sized list containing the total number of vehicles
        """
        return self._cached_counts2

    @property
    def delays(self):
        """Total of vehicles circulating under a velocity threshold

        Returns:
        -------
            * delay: list<int>
            Is a duration sized list containing the total number of slow moving
            vehicles.
        """
        return self._cached_delays2

