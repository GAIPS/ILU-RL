from ilurl.state.elements import Intersection
from ilurl.utils.aux import flatten as flat

class State:

    def __init__(self, network, mdp_params):
        """
        Params:
        ------
        * network: ilurl.networks.base.Network
            network to be described by states

        * mdp_params: ilurl.core.params.MDPParams
            mdp specification: agent, states, rewards, gamma and learning params

        Returns:
        --------

        * state
        """
        # 
        self._tls_ids = network.tls_ids
        # Global features: e.g Time
        # Local features.
        self._intersections = []

        # Local feature
        self._intersections = {
            tls_id: Intersection(mdp_params,
                                 tls_id,
                                 network.tls_phases[tls_id],
                                 network.tls_max_capacity[tls_id])
            for tls_id in network.tls_ids}
            # TODO: replace mdp_params.states --> features

    @property
    def tls_ids(self):
        return self._tls_ids


    def update(self, duration, vehs, tls=None):

        for tls_id, i in self._intersections.items():
            i.update(duration, vehs[tls_id], tls)

    def feature_map(self, filter_by=None, categorize=False, split=False, flatten=False):
        ret = {k:v.feature_map(filter_by=filter_by,
                               categorize=categorize,
                               split=split)
               for k, v in self._intersections.items()}

        if flatten:
            ret = {k: tuple(flat(v)) for k, v in ret.items()}
        return ret
