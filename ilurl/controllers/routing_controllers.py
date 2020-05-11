from operator import itemgetter
from numpy.random import choice

from flow.controllers.base_routing_controller import BaseRouter


class ExpressRouter(BaseRouter):
    """A router used to re-route vehicles in a generic network.

    This class allows the vehicle to pick a `more vacant` route at junctions.

    Usage
    -----
    See base class for usage example.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.choice_edges = []
        self.re_routed = False

    def choose_route(self, env):
        """See parent class."""
        
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))

        veh_pos = vehicles.get_position(veh_id)
        veh_lane = vehicles.get_lane(veh_id)
        next_route = None
        # connection are not in edges but it's valid veh_edge
        if veh_edge in env.network.edges2:
            edges = env.network.edges2
            edge_length = edges[veh_edge]['length']
            
            veh_speed = vehicles.get_speed(veh_id)
            # TODO: adjust by vehicle size and mingap (eta)
            # if edge_length - veh_pos < 10 and veh_edge not in self.choice_edges:
            if edge_length - veh_pos < 10 and not self.re_routed and veh_lane == 0 and \
                    veh_speed > 0.1:

                # if veh_edge == '23738909#2':
                #     print(veh_id, veh_speed, veh_lane)
                #     import pdb
                #     pdb.set_trace()

                veh_choices = len(veh_next_edge)
                # for one approach there might be more than one choice
                if veh_choices > 1:
                    # compute occupancy
                    destination_edges = [
                        (env.network.links[f'{l[0]}_{l[1]}']['to'], l[1])
                        for l in veh_next_edge
                    ]
                    destination_capacity = {
                        dest: edges[dest]['max_capacity'] /
                                    edges[dest]['numLanes']
                        for dest, _ in destination_edges
                    }
                    destination_occupancy = {}
                    for dest, lane in destination_edges:
                        destination_vehicles = [
                           veh for veh in vehicles.get_ids_by_edge(dest)
                           if vehicles.get_lane(veh) == lane
                        ]
                        cap = destination_capacity[dest]
                        destination_occupancy[dest] = \
                           len(destination_vehicles) / cap

                    # both are sufficiently occupied
                    if min(destination_occupancy.values()) > 0.5:
                        # choose the smallest filled route
                        min_edge = \
                            min(destination_occupancy.items(), key=itemgetter(1))
                        next_edge = min_edge[0]

                        #TODO: test if min_edge is already an edge
                        if next_edge not in veh_route:
                            # find routes that contain the min edge
                            sink = veh_route[-1]
                            neighbours = env.network.neighbours_sinks[sink] 

                            # try to find another route which has the same sink
                            # and it contains next_edge
                            routes = [r for r in env.network.routes2[sink] if next_edge in r]
                            if len(routes) == 0 and len(neighbours) > 0:
                                # neightbours search
                                print(f'{veh_id} searching for neighbour')
                                routes = [r for snk in neighbours 
                                            for r in env.network.routes2[snk] if next_edge in r]
                            
                            if len(routes) > 0:
                                route_index = choice(len(routes))
                                chosen_route = routes[route_index]
                                # section of the route made so far
                                new_route = ' '.join(chosen_route).split(next_edge)[-1].split()
                                next_route = tuple([veh_edge, next_edge] + new_route)
                                print(f'{veh_id} re-routing veh_id')

                self.re_routed = next_route is not None
                # if next_route is not None:
                #     self.choice_edges.append(veh_edge)
            return next_route

