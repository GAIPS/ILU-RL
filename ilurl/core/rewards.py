"""Implementation of rewards to be used on state space
    
    TODO: Move from strategy pattern to pure-functional
        implementation
"""
import inspect
from sys import modules

import numpy as np
# from ilurl.core.meta import MetaReward


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
    names, funcs = [], []
    for name, func in inspect.getmembers(this):

        # Is a definition a function
        if inspect.isfunction(func):
            # Is defined in this module
            if inspect.getmodule(func) == this:
                names.append(name)
                funcs.append(func)

    return tuple(names), tuple(funcs)


def build_rewards(mdp_params):
    """High-order function that generates function rewards

    Params:
    ------
    * mdp_params: ilurl.core.params.MDPParams
        mdp specify: agent, states, rewards, gamma and learning params

    Returns:
    --------
    * reward: function(state)
        generates a reward function that receives state as input

    """
    target = mdp_params.reward
    names, funcs = get_rewards()
    if target not in names:
        raise ValueError(f'build_rewards: {target} not in {names}')

    if target == 'reward_max_speed_count':
        # instanciate every scope variable
        target_velocity = mdp_params.target_velocity

        def ret(x):
            return reward_max_speed_count(target_velocity, x)
    else:
        idx = names.index(target)
        fn = funcs[idx]

        def ret(x):
            return fn(x)

    return ret


def reward_max_speed_count(target_velocity, state):
    speeds_counts = state.split()

    ret = {}
    for k, v in speeds_counts.items():
        speeds, counts = v

        if sum(counts) <= 0:
            reward = 0
        else:
            max_cost = \
                np.array([target_velocity] * len(speeds))

            reward = \
                -np.maximum(max_cost - speeds, 0).dot(counts)

        ret[k] = reward
    return ret


def reward_min_delay(states):
    ret = {}
    for tls_id, phase_obs in states.state.items():
        ret[tls_id] = -sum([dly for obs in phase_obs for dly in obs])
    return ret
