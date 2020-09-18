import numpy as np

from ilurl.state.node import Node

def delay(x):
    if x >= 1.0:
        return 0.0
    else:
        return np.exp(-5.0*x)

COUNT_SET = {'average_pressure', 'count', 'speed_score', 'pressure'}

def _check_count(labels):
    return bool(COUNT_SET & set(labels))

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
