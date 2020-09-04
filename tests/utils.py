"""This module provides common functionality among tests"""
from ilurl.utils.aux import flatten

def process_pressure(kernel_data, incoming, outgoing, fctin=1, fctout=1, is_average=False):
    timesteps = list(range(1,60)) + [0]

    ret = 0
    for t, data in zip(timesteps, kernel_data):
        dat = get_veh_locations(data)
        inc = filter_veh_locations(dat, incoming)
        out = filter_veh_locations(dat, outgoing)

        press = len(inc) / fctin - len(out) / fctout
        if is_average:
            ret += round(press, 4)

    if is_average:
        ret = round(ret / 60, 2)
    else:
        ret = round(press, 4)
    return ret


def get_veh_locations(tl_data):
    """Help flattens hierarchial data

    Params:
    ------
        * tl_data: dict<str, dict<int, list<namedtuple<Vehicle>>>>
            nested dict containing tls x phases x vehicles

    Returns:
    --------
        * veh_locations: list<Tuple>
            list containing triplets: veh_id, edge_id, lane
    """

    # 1) Produces a flat generator with 3 informations: veh_id, edge_id, lane
    gen = flatten([(veh.id, veh.edge_id, veh.lane)
                    for ph_data in tl_data.values()
                    for vehs in ph_data.values()
                    for veh in vehs])

    # 2) generates a list of triplets
    it = iter(gen)
    ret = []
    for x in it:
        ret.append((x, next(it), next(it)))
    return ret

def filter_veh_locations(veh_locations, lane_ids):
    """Help flattens hierarchial data"""
    return [vehloc[0] for vehloc in veh_locations if vehloc[1:] in lane_ids]

