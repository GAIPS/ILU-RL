import re
from collections import OrderedDict

import numpy as np

from ilurl.utils.aux import flatten as flat
from ilurl.utils.properties import lazy_property

COUNT_SET = {'average_pressure', 'count', 'speed_score', 'pressure'}

def get_instance_name(x):
    """Gets the name of the instance"""
    return x.__class__.__name__.lower()

def is_unique(xs):
    """Tests if all x in xs belong to the same instance"""
    fn = get_instance_name
    xs_set = {x for x in map(fn, xs)}
    if len(xs_set) == 1:
        return list(xs_set)[0]
    return False

def _check_count(labels):
    return bool(COUNT_SET & set(labels))

def delay(x):
    if x >= 1.0:
        return 0.0
    else:
        return np.exp(-5.0*x)


class Node:
    """Node into the state tree hierarchy

      * Provides a means for bi-direction communication
        between parent and children.
      * Provides bfs function.
      * Implements sequence protocol.
      * Thin-wrapper around domain functionality.
    """
    def __init__(self, parent, node_id, children):
        self.parent = parent
        self.node_id = node_id
        self.children = children
        # Creates an alias
        if children is not None and any(children):
            alias = is_unique(children.values())
            if alias:
                setattr(self, f'{alias}s', children)

    # Sequence protocol
    def __len__(self):
        return len(self.children)

    def __getitem__(self, index):
        return self.children[index]

    # DFS for paths
    def search_path(self, node_id, path, seek_root=True):
        """Returns a path ending on node_ID"""
        # 1) Base case: this is the element
        if node_id == self.node_id:
            return True
        else:
            # 2) Seek root node first.
            if seek_root:
                if self.parent is not None:
                    self.parent.search_path(node_id, path)

            found = False
            # 3) Search down strem
            for chid, ch in self.children.items():
                path.append(chid)
                found = ch.search_path(node_id, path, seek_root=False)
                if found:
                    break
                del path[-1]
            return found



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
        for tls_id, tls in self.intersections.items():
            tls.update(duration, vehs[tls_id], tls)

        # 3) Additional intersection commands
        for tls_id, tls in self.intersections.items():
            tls.after_update()


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


class Intersection(Node):
    """Represents an intersection.

        * Nodes on a transportation network.
        * Root node on a tree hierarchy.
        * Basic unit on the feature map (an agent controls an intersection)
        * Splits outputs.
    """

    def __init__(self, state, mdp_params, tls_id, phases, phase_capacity):
        """Instantiate intersection object

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

        * phase_capacity: dict<int, dict<int, (float, int)>>
            key is the index of the phase.
            tuple is the maximum capacity where:
                float is the max speed of vehicles for the phase.
                float is the max count of vehicles for the phase.

        Returns:
        --------
        * state
            A state is an aggregate of features indexed by intersections' ids.
        """
        # 1) Define children nodes: Phase
        phases = {f'{tls_id}#{phase_id}': Phase(self,
                                                mdp_params,
                                                f'{tls_id}#{phase_id}',
                                                phase_comp,
                                                phase_capacity[phase_id])
                    for phase_id, phase_comp in phases.items()}

        super(Intersection, self).__init__(state, tls_id, phases)


    @property
    def tls_id(self):
        return self.node_id

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
        for pid, phase in self.phases.items():
            # Match last `#'
            # `#' is special character
            *tls_ids, num = pid.split('#')
            phase.update(duration, vehs[int(num)], tls)

    def after_update(self):
        for phase in self.phases.values():
            phase.after_update()

    def reset(self):
        """Clears data from previous cycles, broadcasts method to phases"""
        for phase in self.phases.values():
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
              for phase in self.phases.values()]

        if split:
           return tuple(zip(*ret))
        return ret


class Phase(Node):
    """Represents a phase.

        * Composed of many lanes.
        * Performs categorization.
        * Aggregates wrt lanes.
        * Aggregates wrt time.
    """

    def __init__(self, intersection, mdp_params, phase_id, phase_data, phase_capacity):
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

        # 2) Update lanes
        step_count = 0
        step_delay = 0
        step_flow = set({})
        step_queue = 0
        step_waiting_time = 0
        step_speed = 0
        step_speed_score = 0
        self._update_cached_weight(duration)

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
        self._update_delay(step_delay)
        self._update_flow(step_flow)
        self._update_queue(step_queue)
        self._update_waiting_time(step_waiting_time)
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
        self._cached_delay = 0
        self._cached_flow = set({})
        self._cached_step_flow = set({})

        self._cached_queue = 0
        self._cached_waiting_time = 0
        self._cached_speed = 0
        self._cached_speed_score = 0

        self._cached_weight = 0

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
        w = self._cached_weight
        return round(float(self._cached_delay / (w + 1)), 2)

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
        w = self._cached_weight
        return round(float(self._cached_waiting_time / (w + 1)), 2)


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

    def _update_cached_weight(self, duration):
        """ If duration = 1 then history's weight must be zero i.e
            all weight is given to the new sample"""
        self._cached_weight = int(int(duration) != 1) * (self._cached_weight + 1)

    def _update_count(self, step_count):
        if _check_count(self.labels):
            w = self._cached_weight
            self._cached_count = step_count + (w > 0) * self._cached_count

    def _update_delay(self, step_delay):
        if 'delay' in self.labels:
            w = self._cached_weight
            self._cached_delay = step_delay + (w > 0) * self._cached_delay

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

    def _update_waiting_time(self, step_waiting_time):
        if 'waiting_time' in self.labels:
            w = self._cached_weight
            self._cached_waiting_time = step_waiting_time + (w > 0) * self._cached_waiting_time

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

    def _digitize(self, value, label):
        _bins = self._bins[self._get_derived(label)]
        return int(np.digitize(value, bins=_bins))


class Lane(Node):
    """ Represents a lane within an edge.

        * Leaf nodes.
        * Caches observation data.
        * Performs normalization.
        * Computes feature per time step.
        * Aggregates wrt vehicles.

    """
    def __init__(self, phase, mdp_params, lane_id, max_capacity):
        """ Builds lane.

        Params:
        ------
        * phase: ilurl.state.Phase
            phase associated to the lane (parent node).

        * mdp_params: ilurl.core.params.MDPParams
            mdp specification: agent, states, rewards, gamma and
                               learning params.

        * lane_id: int
            key is the index of the lane.

        * max_capacity: tuple<float, int>
            max velocity a car can travel/lane's max capacity.

        """
        self._min_speed = mdp_params.velocity_threshold
        self._max_capacity = max_capacity
        self._normalize_velocities = mdp_params.normalize_velocities
        self._normalize_vehicles = mdp_params.normalize_vehicles
        self._labels = mdp_params.features
        self.reset()
        super(Lane, self).__init__(phase, lane_id, {})

    @property
    def lane_id(self):
        return self.node_id

    @property
    def labels(self):
        return self._labels

    @property
    def max_vehs(self):
        return self._max_capacity[1]

    @property
    def max_speed(self):
        return self._max_capacity[0]

    @property
    def normalize_velocities(self):
        return self._normalize_velocities

    @property
    def normalize_vehicles(self):
        return self._normalize_vehicles

    def reset(self):
        """ Clears data from previous cycles, defines data structures"""
        self._cached_speed = 0
        self._cached_count = 0
        self._cached_delay = 0
        self._cached_stopped_vehs = 0
        self._cached_flow = set({})

    def update(self, vehs):
        """Update data structures with observation space

            * Holds vehs and tls data for the duration of a cycle

            * Updates features of interest for a time step.

        Params:
        -------
            * vehs: list<namedtuple<ilurl.envs.elements.Vehicles>>
                Container for vehicle data (from VehicleKernel)

        """
        self._update_count(vehs)
        self._update_delay(vehs)
        self._update_stopped_vehs(vehs)
        self._update_flows(vehs)
        self._update_speed(vehs)
        self._update_speed_score(vehs)

    def _update_count(self, vehs):
        """Step update for count variable"""
        if _check_count(self.labels):
            self._cached_count = len(vehs)

    def _update_flows(self, vehs):
        """Step update for flow variable"""
        if 'flow' in self.labels:
            self._cached_flow = {v.id for v in vehs}

    def _update_delay(self, vehs):
        """Step update for delay variable"""
        if 'delay' in self.labels:
            # 1) Speed normalization factor.
            cap = self.max_speed if self.normalize_velocities else 1

            # 2) Compute delay
            step_delays = [delay(v.speed / cap) for v in vehs]
            self._cached_delay = sum(step_delays)

    def _update_stopped_vehs(self, vehs):
        """ Step update for the number of stopped vehicles,
            i.e. the (normalized) number of vehicles with a
            velocity under the defined threshold. """
        if 'queue' in self.labels or 'waiting_time' in self.labels:
            # 1) Normalization factors.
            cap = self.max_speed if self.normalize_velocities else 1 # TODO: Can we remove this and always normalize the speed?
            fct = self.max_vehs if self.normalize_vehicles else 1
            vt = self._min_speed

            # 2) Compute the number of stopped vehicles.
            step_stopped_vehs = [v.speed / cap < vt for v in vehs]

            self._cached_stopped_vehs = sum(step_stopped_vehs) / fct

    def _update_speed(self, vehs):
        """Step update for speed variable"""
        if 'speed' in self.labels:
            # 1) Normalization factor
            cap = self.max_speed if self.normalize_velocities else 1

            # 2) Compute relative speeds:
            # Max prevents relative performance
            step_speeds = [max(self.max_speed - v.speed, 0) / cap for v in vehs]

            self._cached_speed = sum(step_speeds) if any(step_speeds) else 0

    def _update_speed_score(self, vehs):
        """Step update for speed_scores variable"""
        if 'speed_score' in self.labels:
            # 1) Compute speed average
            self._cached_speed_scores = sum([v.speed for v in vehs])

    @property
    def count(self):
        """ Average number of vehicles per time step and lane.

        Returns:
            count: float
        """
        return self._cached_count

    @property
    def delay(self):
        """ The sum of the delay of all vehicles. The delay is defined as a
            function of the vehicles' velocity (see the definition of the
            delay function in the beginning of this file).
        
        Returns:
        -------
            * delay: int
        """
        return self._cached_delay

    @property
    def flow(self):
        """ Unique vehicles present at lane (per lane)
            Returns:
            -------
            * flow: set<string>
                veh_ids
        """
        return self._cached_flow

    @property
    def speed(self):
        """Relative vehicles' speeds per time step and lane.

        * difference between max_speed and observed speed.

        Returns:
            speed: float
        """
        return self._cached_speed

    @property
    def speed_score(self):
        """Sum of raw vehicles' speeds per time step and lane.

        Returns:
            speed_score: float
        """
        return self._cached_speed_scores

    @property
    def stopped_vehs(self):
        """Total of vehicles circulating under a velocity threshold
            per time step and lane.

        Returns:
        -------
            * stopped_vehs: int
                (Normalized) number of vehicles with speed under threshold.
        """
        return self._cached_stopped_vehs
