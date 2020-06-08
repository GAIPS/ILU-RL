"""Implementation of rewards to be used on state space
    
    TODO: Move from strategy pattern to pure-functional
        implementation
"""

import inspect
from sys import modules

import numpy as np
from ilurl.core.meta import MetaReward


def get_rewards():
    """Rewards defined within the module

    * Uses module introspection to get the
      handle for classes.

    Returns:
    -------
    * names: tuple(<str>)
        Names for classes that implement reward computation

    * objects: tuple(<objects>)
        classes wrt camelized names

    Usage:
    -----
    > names, objs = get_rewards()
    > names
    > ('MaxSpeedCountReward', 'MinDelayReward')
    > objs
    > (<class 'ilurl.core.rewards.MaxSpeedCountReward'>,
       <class 'ilurl.core.rewards.MinDelayReward'>)
    """
    this = modules[__name__]
    names, objects = [], []
    for name, obj in inspect.getmembers(this):

        # Is a definition a class?
        if inspect.isclass(obj):
            # Is defined in this module
            if inspect.getmodule(obj) == this:
                names.append(name)
                objects.append(obj)

    return tuple(names), tuple(objects)


def build_rewards(mdp_params):
    """Builder that defines all rewards

    Params:
    ------
    * mdp_params: ilurl.core.params.MDPParams
        mdp specify: agent, states, rewards, gamma and learning params

    Returns:
    --------
    * reward: ilurl.core.rewards.XXXReward
        an instance of reward object

    """
    target = mdp_params.reward
    rewnames, rewclasses = get_rewards()
    if target not in rewnames:
        raise ValueError(f'build_rewards: {target} not in {rewnames}')

    idx = rewnames.index(target)
    reward_cls = rewclasses[idx]

    return reward_cls(mdp_params)


class MaxSpeedCountReward(object, metaclass=MetaReward):

    def __init__(self,  mdp_params):
        """Creates a reference to the input state"""
        if not hasattr(mdp_params, 'target_velocity'):
            raise ValueError('MDPParams must define target_velocity')
        else:
            self.target_velocity = mdp_params.target_velocity

    def calculate(self, state):
        speeds_counts = state.split()

        ret = {}
        for k, v in speeds_counts.items():
            speeds, counts = v

            if sum(counts) <= 0:
                reward = 0
            else:
                max_cost = \
                    np.array([self.target_velocity] * len(speeds))

                reward = \
                    -np.maximum(max_cost - speeds, 0).dot(counts)

            ret[k] = reward
        return ret


class MinDelayReward(object, metaclass=MetaReward):
    def __init__(self, *args, **kwargs):
        pass

    def calculate(self, states):
        ret = {}
        for tls_id, phase_obs in states.state.items():
            ret[tls_id] = -sum([dly for obs in phase_obs for dly in obs])
        return ret


# TODO: implement Rewards definition 2
class MinDeltaDelayReward(object, metaclass=MetaReward):
    def __init__(self, state):
        """Creates a reference to the input state"""
        if state.__class__ != 'DeltaDelay':
            raise ValueError(
                'MinDeltaDelay reward expects `DeltaDelay` state')

    def calculate(self):
        return -sum(self._state.to_list())
