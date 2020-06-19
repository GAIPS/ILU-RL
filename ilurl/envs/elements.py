'''
    Elements or parts of simulated environments

'''
from collections import namedtuple

Vehicle = namedtuple('Vehicle', 'id tls_id edge_id lane speed pos')
TrafficLightSignal = namedtuple('TrafficLight', 'tls_id state')


def build_vehicles(node_id, components, veh_kernel):
    """Builds Vehicle Element

        * Definition of vehicle is everything the state
          classes need to know -- for computing the state.

        * Extend namedtuple instead of extending input
          values from state's update method.

        * Thin wrapper around vehicles kernel.

        Params:
        ------
        * node_id: str
            Usually node_id <=> tls_id

        * components: tuple<str, int>
            edge_id, lane

        * veh_kernel: flow.core.kernel.vehicle
            See definition

        Returns:
        -------
        * vehs: list<namedtuple<Vehicle>>
    """
    vehs = []
    for component in components:
        edge_id, lanes = component

        veh_ids = [veh_id for veh_id in veh_kernel.get_ids_by_edge(edge_id)
                   if veh_kernel.get_lane(veh_id) in lanes]

        for veh_id in veh_ids:
            veh = Vehicle(
                veh_id,
                node_id,
                edge_id,
                veh_kernel.get_lane(veh_id),
                veh_kernel.get_speed(veh_id),
                veh_kernel.get_position(veh_id)
            )
            vehs.append(veh)

    vehs = sorted(vehs, key=lambda x: x.id)
    return vehs



