from copy import deepcopy

from ilurl.state.node import Node
from ilurl.state.phase import Phase

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
        # 1) Deepcopy categories
        _mdp_params = deepcopy(mdp_params)
        if _mdp_params.discretize_state_space:
            _mdp_params.categories = _mdp_params.categories[tls_id]

        # 1) Define children nodes: Phase
        phases = {f'{tls_id}#{phase_id}': Phase(self,
                                                _mdp_params,
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


