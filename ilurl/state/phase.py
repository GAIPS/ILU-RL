import re
from collections import OrderedDict
import numpy as np

from ilurl.state.node import Node
from ilurl.state.lane import Lane

from ilurl.utils.properties import lazy_property

COUNT_SET = {'average_pressure', 'count', 'speed_score', 'pressure'}

def _check_count(labels):
    return bool(COUNT_SET & set(labels))

def uniq(tl_state):
    """Unique that preservers order"""
    ret = [s for s in tl_state.upper()]
    ret = OrderedDict.fromkeys(ret)
    ret = [r for r in ret]
    return ret

class Phase(Node):
    """Represents a phase.

        * Composed of many lanes.
        * Performs categorization.
        * Aggregates wrt lanes.
        * Aggregates wrt time.
    """

    def __init__(self, intersection, mdp_params, phase_id, phase_order, phase_data, phase_capacity):
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

        * phase_capacity: dict<int, dict<int, tuple<float, int>>
            key: int phase_id
            key: int lane_number
            tuple<float, int>
            max speed of vehicles for the phase.
            max count of vehicles for the phase.

        """
        # 1) Define base attributes
        self._labels = mdp_params.features
        self._matcher = re.compile('\[(.*?)\]')
        self._lagged = any('lag' in lbl for lbl in mdp_params.features)
        self._normalize_velocities = mdp_params.normalize_velocities
        self._normalize_vehicles = mdp_params.normalize_vehicles
        self._order = phase_order

        # 2) Get categorization bins.
        # fn: extracts category_<feature_name>s from mdp_params
        def fn(x):
            z = self._get_derived(x)
            return [getattr(mdp_params, y) for y in dir(mdp_params)
                    if (z in y) and ('category_' in y)][0]
        self._bins = {_feat: fn(_feat) for _feat in mdp_params.features}


        # 3) Instantiate lanes
        lanes = {}
        for incoming in phase_data['incoming']:
            edge_id, lane_nums = incoming
            for lane_num in lane_nums:
                lane_id = (edge_id, lane_num)
                lanes[lane_id] = Lane(self, mdp_params, lane_id, phase_capacity[lane_num])

        # 4) Save outgoing lane ids.
        outgoing_ids = []
        for outgoing in phase_data['outgoing']:
            edge_id, lane_nums = outgoing
            for lane_num in lane_nums:
               outgoing_ids.append((edge_id, lane_num))
        self._outgoing_ids = outgoing_ids

        self.cached_features = {}
        self._phase_order = phase_order
        assert phase_order < 2
        super(Phase, self).__init__(intersection, phase_id, lanes)


    @property
    def phase_id(self):
        return self.node_id

    @property
    def labels(self):
        return self._labels

    @property
    def lagged(self):
        return self._lagged

    @property
    def order(self):
        return self._phase_order

    @property
    def normalize_velocities(self):
        return self._normalize_velocities

    @property
    def normalize_vehicles(self):
        return self._normalize_vehicles

    @property
    def incoming(self):
        """Alias to lanes or incoming approaches."""
        return self.lanes

    @lazy_property
    def outgoing(self):
        """Defines outgoing lanes as incoming lanes from other agents."""
        # 1) Search for outgoing lanes.
        # edge_ids from outgoing lanes which are 
        paths = []
        outgoing_ids = self._outgoing_ids
        for outgoing_id in outgoing_ids:
            path = []
            self.search_path(outgoing_id, path)
            paths.append(path)

        # 2) Get a reference for the edge
        ret = {}
        state = self.parent.parent
        for path in paths:
            if any(path):
               tls_id, phase_id, lane_id = path
               ret[lane_id] = state[tls_id][phase_id][lane_id]

        return ret

    @lazy_property
    def max_speed(self):
        """Phase Max. Speed

        Consolidates
            * max_speed is an attribute from the inbound lanes not from phase.
            * speed_score needs a phase max_speed
        """
        return max([inc.max_speed for inc in self.incoming.values()])

    @lazy_property
    def max_vehs(self):
        """Phase Max. Capacity wrt incoming lanes

        Consolidates
            * max_capacity: is the sum of all incoming lanes' capacities
        """
        return sum([inc.max_vehs for inc in self.incoming.values()])

    @lazy_property
    def max_vehs_out(self):
        """Phase Max. Capacity wrt outgoing lanes

        Consolidates
            * max_capacity: is the sum of all outgoing lanes' capacities
        """
        return sum([out.max_vehs for out in self.outgoing.values()])

    def get_index(self, tls):
        """Index within representation

        zero for green and  one for red

        Params:
        -------
        state_string: str
            ex: 'GGGrrrrrGG','rrrrrGGGGG'

        Returns:
        --------
        index: int {0, 1}
            the state string
        """
        sig = uniq(tls)[self.order]
        return int((sig == 'G') * 0 + (sig == 'R') * 1)

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
        # 1) Define a helpful filtering function.
        def _in(veh, lane):
            return (veh.edge_id, veh.lane) == lane.lane_id

        # 3) 0 is green and 1 is red
        ind = self.get_index(tls)

        # 2) Update lanes
        step_count = 0
        step_delay = 0
        step_flow = set({})
        step_queue = 0
        step_waiting_time = 0
        step_speed = 0
        step_speed_score = 0
        self._update_cached_weight(ind)

        for lane in self.lanes.values():
            _vehs = [v for v in vehs if _in(v, lane)]
            lane.update(_vehs)

            step_count += lane.count if _check_count(self.labels) else 0
            step_delay += lane.delay if 'delay' in self.labels else 0
            step_flow = step_flow.union(lane.flow) if 'flow' in self.labels else 0
            step_queue = max(step_queue, lane.stopped_vehs) if 'queue' in self.labels else 0
            step_waiting_time += lane.stopped_vehs if 'waiting_time' in self.labels else 0
            step_speed += lane.speed if 'speed' in self.labels else 0
            step_speed_score += lane.speed_score if 'speed_score' in self.labels else 0

        self._update_count(step_count)
        self._update_delay(step_delay, ind)
        self._update_flow(step_flow)
        self._update_queue(step_queue)
        self._update_waiting_time(step_waiting_time, ind)
        self._update_speed(step_speed)
        self._update_speed_score(step_speed_score)

        # 3) Stores previous cycle for lag labels.
        self._update_lag(duration)

    def after_update(self):
        """Commands to be executed once after every phase has been updated"""
        if 'average_pressure' in self.labels:
            w = self._cached_weight
            self._cached_average_pressure = self.pressure + (w > 0) * self._cached_average_pressure

    def reset(self):
        """Clears data from previous cycles, broadcasts method to lanes"""
        # 1) Communicates update for every lane
        for lane in self.lanes.values():
            lane.reset()

        # 2) Erases previous cycle's memory
        # Two memories are necessary:
        # (i) store values for next cycle.
        # (ii) preserve values for current cycle.
        self._cached_store = {}
        self._cached_apply = {}


        # 3) Defines or erases history
        self._cached_average_pressure = 0
        self._cached_count = 0
        self._cached_delay = [0.0, 0.0]
        self._cached_flow = set({})
        self._cached_step_flow = set({})

        self._cached_queue = 0
        self._cached_waiting_time = [0.0, 0.0]
        self._cached_speed = 0
        self._cached_speed_score = 0

        self._cached_weight = [0.0, 0.0]
        self._last_index = None

    def feature_map(self, filter_by=None, categorize=False):
        """Computes phases' features

            This method must be called in every cycle
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
    def average_pressure(self):
        """Pressure controller

        Difference of number of vehicles in incoming or outgoing lanes.

        Reference:
        ---------
        * Wade Genders and Saiedeh Razavi, 2019
            An Open Source Framework for Adaptive Traffic Light Signal Control.

        See also:
        --------
        * Wei, et al, 2018
            PressLight: Learning Max Pressure Control to Coordinate Traffic
                        Signals in Arterial Network.

        * Pravin Varaiya, 2013
            Max pressure control of a network of signalized intersections

        """
        w = self._cached_weight
        ret = round(self._cached_average_pressure / (w + 1), 2)
        return ret


    @property
    def count(self):
        """Aggregates count wrt time and lanes

        Returns:
        -------
        * count: float
            The average number of vehicles in the approach
        """
        w = self._cached_weight
        ret = round(float(self._cached_count / (w + 1)), 2)
        return ret

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
        weights_delays = zip(self._cached_weight, self._cached_delay)
        ret = []
        for w, dly in weights_delays:
            ret.append(round(float(dly / (w + 1)), 2))
        return ret

    @property
    def flow(self):
        """Vehicles dispatched
        Returns:
        -------
        * flow: int
           The number of vehicles dispatched during the cycle.
        References:
        ----------
        """
        ret = round(float(len(self._cached_flow - self._cached_step_flow)), 2)
        return ret


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
        return round(float(self._cached_queue), 2)

    @property
    def waiting_time(self):
        """Max. queue of vehicles wrt lane and time steps.

        Returns:
        -------
        * waiting_time: float
            The average number of cars under a given velocity threshold.

        """
        waits = zip(self._cached_weight, self._cached_waiting_time)
        ret = []
        for w, wt in waits:
            ret.append(round(float(wt / (w + 1)), 2))
        return ret


    @property
    def pressure(self):
        """Pressure controller

        * Difference of number of vehicles in incoming or outgoing lanes.
        * Doesn't take into account.

        Reference:
        ---------
        * Wade Genders and Saiedeh Razavi, 2019
            An Open Source Framework for Adaptive Traffic Light Signal Control.

        See also:
        --------
        * Wei, et al, 2018
            PressLight: Learning Max Pressure Control to Coordinate Traffic
                        Signals in Arterial Network.

        * Pravin Varaiya, 2013
            Max pressure control of a network of signalized intersections

        """
        fct = self.max_vehs if self.normalize_vehicles else 1
        inbound = self._compute_pressure(self.incoming) / fct

        fct = self.max_vehs_out if self.normalize_vehicles else 1
        outbound = self._compute_pressure(self.outgoing) / fct
        return round(float(inbound - outbound), 4)


    def _compute_pressure(self, lanes):
        counts = [_lane.count for _lane in lanes.values()]
        return np.sum(counts).round(4)

    @property
    def speed(self):
        """Aggregates speed wrt time and lane

        Returns:
        -------
        * speed: float
            The average speed of all cars in the phase
        """
        if self._cached_count > 0:
            ret = float(self._cached_speed / self._cached_count)
            return round(ret, 2)
        else:
            return 0.0

    @property
    def speed_score(self):
        """Aggregates speed wrt time and lane

        Returns:
        -------
        * speed: float
            The average speed of all cars in the phase
        """
        if self._cached_count > 0:
            ret = min(float(self._cached_speed_score / (self._cached_count * self.max_speed)), 1)
            return round(ret, 2)
        else:
            return 0.0

    def _update_cached_weight(self, ind):
        # When it switches to another color resets current
        self._cached_weight[ind] = \
            int(self._last_index == ind) * (self._cached_weight[ind] + 1)

        self._last_index = ind

    def _update_count(self, step_count):
        if _check_count(self.labels):
            w = self._cached_weight
            self._cached_count = step_count + (w > 0) * self._cached_count

    def _update_delay(self, step_delay, ind):
        if 'delay' in self.labels:
            w = self._cached_weight[ind]
            self._cached_delay[ind] = \
                step_delay + (w > 0) * self._cached_delay[ind]

    def _update_flow(self, step_flow):
        if 'flow' in self.labels:
            w = self._cached_weight
            if w > 0:
                # union w/ previous' state step flow
                self._cached_flow = self._cached_flow.union(self._cached_step_flow)
            else:
                # assign to previous' state step flow
                self._cached_flow = self._cached_step_flow
            # this is the current vehicles on lanes.
            self._cached_step_flow = step_flow

    def _update_queue(self, step_queue):
        if 'queue' in self.labels:
            w = self._cached_weight
            self._cached_queue = max(step_queue, (w > 0) * self._cached_queue)

    def _update_waiting_time(self, step_waiting_time, ind):
        if 'waiting_time' in self.labels:
            w = self._cached_weight[ind]
            self._cached_waiting_time[ind] = \
                step_waiting_time + (w > 0) * self._cached_waiting_time[ind]

    def _update_speed(self, step_speed):
        if 'speed' in self.labels:
            w = self._cached_weight
            self._cached_speed = step_speed + (w > 0) * self._cached_speed

    def _update_speed_score(self, step_speed_score):
        if 'speed_score' in self.labels:
            w = self._cached_weight
            m = self._cached_speed_score
            self._cached_speed_score = step_speed_score + (w > 0) * m

    def _update_lag(self, duration):
        """`rotates` the cached features
            (i) previous cycle's stored features go to apply.
            (ii) current cycle's features go to store.
        """
        if duration == 0 and self.lagged:
            derived_features = [self._get_derived(lbl) for lbl in self.labels if 'lag' in lbl]
            self._cached_apply.update(self._cached_store)
            self._cached_store = {f: getattr(self, f) for f in derived_features}

    def _get_feature_by(self, label):
        """Returns feature by label"""
        if 'lag' in label:
            derived_feature = \
                self._matcher.search(label).groups()[0]
            return self._cached_apply.get(derived_feature, 0.0)
        return getattr(self, label)

    def _get_derived(self, label):
        """Returns label or derived label"""
        derived_label = None
        if 'lag' in label:
            derived_label = \
                self._matcher.search(label).groups()[0]
        return derived_label or label

    def _digitize(self, values, label):
        _bins = self._bins[self._get_derived(label)]
        return [int(np.digitize(val, bins=_bins)) for val in values]
