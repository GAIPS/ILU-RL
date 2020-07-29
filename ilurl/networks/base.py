"""This module acts as a wrapper for networks generated from network data"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-10'

import operator as op
from itertools import groupby
from copy import deepcopy
from collections import defaultdict

from flow.core.params import InitialConfig, TrafficLightParams
from flow.core.params import (VehicleParams, SumoCarFollowingParams,
                            SumoLaneChangeParams)
from flow.core.params import NetParams
import flow.networks.base as flownet

from ilurl.utils.properties import lazy_property
from ilurl.params import InFlows
from ilurl.loaders.nets import (get_routes, get_edges, get_path,
                                get_logic, get_connections, get_nodes,
                                get_types, get_tls_custom)

#from ilurl.controllers.routing_controllers import GreedyRouter


class Network(flownet.Network):
    """This class leverages on specs created by SUMO."""

    def __init__(self,
                 network_id,
                 horizon=360,
                 net_params=None,
                 vehicles=None,
                 demand_type='constant',
                 initial_config=None,
                 tls_type='rl'):

        """Builds a new network from inflows -- the resulting
        vehicle trips will be stochastic use it for training"""
        self.network_id = network_id

        baseline = (tls_type == 'actuated')
        self.cycle_time, self.programs = get_tls_custom(
                                network_id, baseline=baseline)

        if initial_config is None:
            initial_config = InitialConfig(
                edges_distribution=tuple(get_routes(network_id).keys())
            )

        if net_params is None:
            #TODO: check vtype
            if vehicles is None:
                vehicles = VehicleParams()
                vehicles.add(
                    veh_id="human",
                    car_following_params=SumoCarFollowingParams(
                        min_gap=2.5,
                        decel=7.5,  # Avoid collisions at emergency stops.
                    ),
                    lane_change_params=SumoLaneChangeParams(
                        lane_change_mode='strategic'
                    )
                )

            inflows = InFlows(network_id,
                              horizon,
                              demand_type,
                              initial_config=initial_config)

            net_params = NetParams(inflows,
                                   template=get_path(network_id, 'net'))

        # static program (required for rl)
        tls_logic = TrafficLightParams(baseline=False)
        if tls_type not in ('actuated', ):
            programs = get_logic(network_id)
            if programs:
                for prog in programs:
                    node_id = prog.pop('id')
                    prog['tls_type'] = prog.pop('type')
                    prog['programID'] = int(prog.pop('programID')) + 1
                    tls_logic.add(node_id, **prog)
        else:
            for tls_id, tls_args in self.programs.items():
                tls_logic.add(tls_id, **tls_args)

        super(Network, self).__init__(
                 network_id,
                 vehicles,
                 net_params,
                 initial_config=initial_config,
                 traffic_lights=tls_logic
        )

        self.nodes = self.specify_nodes(net_params)
        self.edges = self.specify_edges(net_params)
        self.connections = self.specify_connections(net_params)
        self.types = self.specify_types(net_params)


    def specify_nodes(self, net_params):
        return get_nodes(self.network_id)


    def specify_edges(self, net_params):
        return self._add_edges_capacity(
            get_edges(self.network_id)
        )


    def specify_connections(self, net_params):
        """Connections bind edges' lanes to one another at junctions
         
            DEF:
            ----
            definitions follow the standard
            *Name   :Type
                Description

            *from   :edge id (string)
                The ID of the incoming edge at which the connection
                begins
            *to     :edge id (string)
                The ID of the outgoing edge at which the connection ends
            *fromLane   :index (unsigned int)
                The lane of the incoming edge at which the connection
                begins
            *toLane     :index (unsigned int)
                The lane of the outgoing edge at which the connection ends
            *via    :lane id (string)
                The id of the lane to use to pass this connection across the junction
            *tl     :traffic light id (string
                The id of the traffic light that controls this connection; the attribute is missing if the connection is not controlled by a traffic light
            *linkIndex  :index (unsigned int
                The index of the signal responsible for the connection within the traffic light; the attribute is missing if the connection is not controlled by a traffic light
            *dir:enum
                ("s" = straight, "t" = turn, "l" = left, "r" = right, "L" = partially left, R = partially right, "invalid" = no direction
            The direction of the connection
            *state:enum
                ("-" = dead end, "=" = equal, "m" = minor link, "M" = major link, traffic light only: "O" = controller off, "o" = yellow flashing, "y" = yellow minor link, "Y" = yellow major link, "r" = red, "g" = green minor, "G" green major
            The state of the connection

            REF:
            ----
            http://sumo.sourceforge.net/userdoc/Networks/SUMO_Road_Networks.html
        """
        return get_connections(self.network_id)


    def specify_routes(self, net_params):
        return get_routes(self.network_id)

    def specify_types(self, net_params):
        return get_types(self.network_id)

    def specify_edge_starts(self):
        "see parent class"
        return [(e['id'], e['length']) for e in get_edges(self.network_id)]

    @lazy_property
    def links(self):
        """Dict version from connections"""
        conns = deepcopy(self.connections)
        return {conn.pop('via'): conn for conn in conns if 'via' in conn} 

    @lazy_property
    def edges2(self):
        """Edges as dictionary instead of a list"""
        return {data['id']: {k:v for k, v in data.items() if k != 'id'}
                for data in self.edges}

    @lazy_property
    def routes2(self):
        """Sinks to routes dictionary --  no prob. emissions

          Returns:
            routes: dict<<str>, list<str>>
                key: str .: edge_id routes' sink
                values: list .: edge_ids
        """
        routes2 = defaultdict(list)
        for src, routes_weights in get_routes(self.network_id).items():
            for route, weights in routes_weights:
                routes2[route[-1]].append(route)
        return routes2

    @lazy_property
    def neighbours_sinks(self):
        """Sinks to routes dictionary --  no prob. emissions

          Returns:
            routes: dict<<str>, list<str>>
                key: str .: edge_id routes' sink
                values: list .: edge_ids
        """
        edges = self.edges2
        nodes = self.nodes
        sinks = self.routes2.keys()
        neighbours = {}

        for sink in sinks:
            sink_node = edges[sink]['from']

            # those neightbours share the same junction
            neighbours[sink] = [eid for eid, data in edges.items()
                                 if data['from'] == sink_node and
                                    eid != sink and eid in sinks]

            # those neighbours are on adjacent junctions
            adjacent_nodes = [node['id'] for node in nodes for data in edges.values()
                              if data['from'] == node['id'] and data['to'] == sink_node]

            neighbours[sink] += [eid for eid in sinks
                                     if edges[eid]['from'] in adjacent_nodes]
        return neighbours

    @lazy_property
    def tls_incoming(self):
        """Returns the incoming approaches for a traffic light junction

        Params:
        ------
        * nodeid: string
            a valid nodeid in self.nodes

        Returns:
        ------
        * approaches: dict<string, list<string>>
            list of mappings from node_id -> incoming edge ids

        Usage:
        -----
         > network.tls_incoming
         > {'247123161': ['-238059324', '-238059328', '309265401', '383432312']}

        DEF:
        ---
        A roadway meeting at an intersection is referred to as an approach.
        At any general intersection, there are two kinds of approaches:
        incoming approaches and outgoing approaches.

        An incoming approach is one on which cars can enter the intersection.

        REF:
        ---
            * Wei et al., 2019
            http://arxiv.org/abs/1904.08117
        """
        return {nid: [e['id'] for e in self.edges if e['to'] == nid]
                for nid in self.tls_ids}

    @lazy_property
    def tls_outgoing(self):
        """Returns the outgoing approaches for a traffic light junction

        Params:
        ------
        * nodeid: string
            a valid nodeid in self.nodes

        Returns:
        ------
        * approaches: dict<string, list<string>>
            list of mappings from node_id -> outgoing edge ids

        Usage:
        -----
         > network.tls_outgoing
         > {'247123161': ['-238059324', '-238059328', '309265401', '383432312']}

        DEF:
        ---
        A roadway meeting at an intersection is referred to as an approach.
        At any general intersection, there are two kinds of approaches:
        incoming approaches and outgoing approaches.

        An outgoing approach is one on which cars can exit the intersection.

        REF:
        ---
            * Wei et al., 2019
            http://arxiv.org/abs/1904.08117
        """
        return {nid: [e['id'] for e in self.edges if e['from'] == nid]
                for nid in self.tls_ids}

    @lazy_property
    def tls_phases(self):
        """Returns a nodeid x sets of non conflicting movement patterns.
            The sets are index by integers and the moviment patterns are
            expressed as lists of approaches. We consider only incoming
            approaches to be controlled by phases.

        Returns:
        ------
        * phases: dict<string,dict<int, dict<string, obj>>>
            keys: nodeid, phase_id, 'states', 'components'

        Usage:
        -----
        > network.tls_states
        > {'gneJ2':
            ['GGGgrrrrGGGgrrrr', 'yyygrrrryyygrrrr', 'rrrGrrrrrrrGrrrr',
            'rrryrrrrrrryrrrr', 'rrrrGGGgrrrrGGGg', 'rrrryyygrrrryyyg',
            'rrrrrrrGrrrrrrrG', 'rrrrrrryrrrrrrry']}

        > network.tls_phases
        > {'gneJ2':
            {0: {'components':
                    [('-gneE8', [0, 1, 2]), ('gneE12', [0, 1, 2])],
                    'states': ['GGGgrrrrGGGgrrrr']
                },
             1: {'components':
                     [('-gneE8', [2]), ('gneE12', [2])],
                  'states': ['yyygrrrryyygrrrr', 'rrrGrrrrrrrGrrrr',
                             'rrryrrrrrrryrrrr']
                },
             2: {'components':
                     [('gneE7', [0, 1, 2]), ('-gneE10', [0, 1, 2])],
                 'states': ['rrrrGGGgrrrrGGGg']
                 },
             3: {'components':
                     [('gneE7', [2]), ('-gneE10', [2])],
                 'states': ['rrrryyygrrrryyyg', 'rrrrrrrGrrrrrrrG',
                            'rrrrrrryrrrrrrry']
                }
             }
           }
        DEF:
        ---
        A phase is a combination of movement signals which are
        non-conflicting. The relation from states to phases is
        such that phases "disregards" yellow configurations
        usually num_phases = num_states / 2

        REF:
        ---
        Wei et al., 2019
        http://arxiv.org/abs/1904.08117
        """

        ret = defaultdict(dict)
        def fn(x, n):
            return x.get('tl') == n and 'linkIndex' in x

        for nid in self.tls_ids:
            # green and yellow are considered to be one phase
            connections = [c for c in self.connections if fn(c, nid)]
            states = self.tls_states[nid]
            incoming_links = {
                int(cn['linkIndex']): (cn['from'], int(cn['fromLane']))
                for cn in connections if 'linkIndex' in cn
            }

            outgoing_links = {
                int(cn['linkIndex']): (cn['to'], int(cn['toLane']))
                for cn in connections if 'linkIndex' in cn
            }
            i = 0
            components = {}
            for state in states:
                # # components: (linkIndex, phase_num, edge_id, lane)
                # # states are indexed by linkIndex
                # # 'G', 'g' states are grouped together.
                # components = {
                #     (lnk,) + edge_lane
                #     for lnk, edge_lane in links.items()
                #     if state[lnk] in ('G','g')
                # }

                # # adds components if they don't exist
                # if components:
                #     found = False
                #     # sort by link, edge_id
                #     components = \
                #         sorted(components, key=op.itemgetter(0, 1))

                #     # groups lanes by edge_ids and states
                #     components = \
                #         [(k, list({l[-1] for l in g}))
                #          for k, g in groupby(components, key=op.itemgetter(1))]
                incoming = self._group_links(incoming_links, state)
                outgoing = self._group_links(outgoing_links, state)

                # Match states
                # The state related to this phase might already have been added.
                found = False
                if any(incoming) or any(outgoing):
                    for j in range(0, i + 1):

                        if j in ret[nid]:
                            # same edge_id and lanes
                            found = ret[nid][j]['incoming'] == incoming and \
                                        ret[nid][j]['outgoing'] == outgoing

                            if found:
                                ret[nid][j]['states'].append(state)
                                break

                    if not found:
                        ret[nid][i] = {
                            'incoming': incoming,
                            'outgoing': outgoing,
                            'states': [state]
                        }
                        i += 1
                else:
                    # add to the last phase
                    # states only `r` and `y`
                    ret[nid][i - 1]['states'].append(state)
        return ret

    def _group_links(self, links, state):
        """Transforms links into components

        Parameters:
        ----------
        * links: dict<int, tuple<str, int>
              links: linkIndex --> (edge_id, lane)
              ex: {7: ('-309265400', 0), 8: ('-309265400', 0), ..., 6:('309265402', 1)}

        * state: str
           ex: 'rrrrrGGGGG'

        Returns:
        --------
            component: list<tuple<str,list<int>>>
                    component --> [(edge_a, [lane_0, lane_1]), ...]
                    ex: [('309265401', [0, 1]), ('-238059328', [0, 1])]
        """
        # components: (linkIndex, phase_num, edge_id, lane)
        # states are indexed by linkIndex
        # 'G', 'g' states are grouped together.
        components = {
            (lnk,) + edge_lane
            for lnk, edge_lane in links.items()
            if state[lnk] in ('G','g')
        }
        # adds components if they don't exist
        if components:
            # sort by link, edge_id
            components = \
                sorted(components, key=op.itemgetter(0, 1))

            # groups lanes by edge_ids and states
            components = \
                [(k, list({l[-1] for l in g}))
                 for k, g in groupby(components, key=op.itemgetter(1))]
        return components

    @lazy_property
    def tls_max_capacity(self):
        """Max speeds and counts that an intersection can handle

        Returns:
        -------
            * max_capacity: dict<string, tuple<float, float>>
                keys: tls_id
                float: max. speeds (m/s) or counts (vehs)

        Usage:
        > network.tls_max_capacity
        > {'247123161': {0:{(22.25, 40), 1: (7.96, 12)}

        """
        max_capacity = {}
        for tls_id in self.tls_ids:
            _max_capacity = {}
            for phase, data in self.tls_phases[tls_id].items():
                max_count, max_speed = 0, 0
                for edge_id, lanes in data['components']:
                    edge = [e for e in self.edges if e['id'] == edge_id][0]
                    k = len(lanes) / edge['numLanes']
                    max_count += edge['max_capacity'] * k
                    max_speed = max(edge['max_speed'], max_speed)
                _max_capacity[phase] = (max_speed, max_count)
            max_capacity[tls_id] = _max_capacity
        return max_capacity

    @lazy_property
    def tls_states(self):
        """states wrt to programID = 1 for traffic light nodes

        Returns:
        -------
            * states: dict<string, list<string>>

        Usage:
        ------
        > network.tls_states
        > {'247123161': ['GGrrrGGrrr', 'yyrrryyrrr', 'rrGGGrrGGG', 'rryyyrryyy']}O

        REF:
        ----
            http://sumo.sourceforge.net/userdoc/Simulation/Traffic_Lights.html
        """
        cfg = self.traffic_lights.get_properties()

        def fn(x):
            return x['type'] in ('static', 'actuated') and x['programID'] == 1

        return {
            tid: [p['state'] for p in cfg[tid]['phases'] if fn(cfg[tid])]
            for tid in self.tls_ids
        }

    @lazy_property
    def tls_durations(self):
        """Gives the times or durations in seconds for each of the states
        on the default programID = 1

        Returns:
        -------
            * durations: list<int>
                a list of integer representing the time in seconds, each
                state is alloted on the default static program.

        Usage:
        ------
        > network.tls_durations
        > {'247123161': [39, 6, 39, 6]}

        REF:
        ----
            http://sumo.sourceforge.net/userdoc/Simulation/Traffic_Lights.html
        """
        cfg = self.traffic_lights.get_properties()

        def fn(x):
            return x['type'] == 'static' and x['programID'] == 1

        return {
            t: [int(p['duration']) for p in cfg[t]['phases'] if fn(cfg[t])]
            for t in self.tls_ids
        }

    @lazy_property
    def tls_ids(self):
        """List of nodes which are also traffic light signals

        Returns
        -------
            * nodeids: list<string>
        Usage:
        -----
        # intersection
        > network.tls_ids
        ['247123161']

        """
        return [n['id'] for n in self.nodes if n['type'] == 'traffic_light']

    @lazy_property
    def phases_per_tls(self):
        """Dict containing the number of phases per traffic light.

        Returns
        -------
            * phases_per_tls: dict

        """
        return {tid: len(self.tls_phases[tid])
                    for tid in self.tls_ids}

    @lazy_property
    def num_signal_plans_per_tls(self):
        """Dict containing the number of signal plans available
        at 'programs' per traffic light.

        Returns
        -------
            * num_signal_plans_per_tls: dict

        """
        return {tid: len(self.programs[tid]) 
                    for tid in self.tls_ids}

    def _add_edges_capacity(self, edges):
        """Updates edges by providing capacity as the max density number of cars
            per edge
        
        Limitations:
        -----------
        * It considers an average number vehicles over all vehicle_types
        * If vehicle length is not provided converts it to length 5 default

        Sumo:
        -----
        length 	float 	5.0 	The vehicle's netto-length (length) (in m)
        minGap 	float 	2.5 	Empty space after leader [m]
        maxSpeed 	float 	55.55 (200 km/h) for vehicles   The vehicle's maximum velocity (in m/s)

        Use case:
        --------
         Determine the theoretical flow:
         q (flow) [cars/h]  = D (density) [cars/km] x V (speed) [km/h]

        References:
        -----------
        https://en.wikipedia.org/wiki/Fundamental_diagram_of_traffic_flow#Basic_statements
        http://sumo.sourceforge.net/userdoc/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#available_vtype_attributes
        """
        # Summarize over vehicle types
        xs, vs = 0, 0
        for i, veh_type in enumerate(self.vehicles.types):
            # compute the average vehicle length
            x = veh_type.get('minGap', 2.5) + veh_type.get('length', 5.0)
            v = veh_type.get('maxSpeed', 55.55)
            xs = (x + i * xs) / (i + 1)     # mean of lengths
            vs = (v + i * vs) / (i + 1)     # mean of max_speeds

        # Apply over edges
        for edge in edges:
            edge['max_capacity'] = int(edge['length'] / xs) * edge['numLanes']
            # max of mean speeds (max_speed is too conservative)
            edge['max_speed'] = 0.5 * edge.get('speed', vs)
            
        return edges
