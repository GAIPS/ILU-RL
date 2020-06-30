"""Implementation of rewards to be used on state space
    
    TODO: Move from strategy pattern to pure-functional
        implementation
"""
import pdb
import inspect
from sys import modules

import numpy as np

QUEUE = {}


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

        def ret(x, *args):
            return reward_max_speed_count(target_velocity, x, *args)

    elif target == 'reward_min_queue_squared':

        def ret(x, *args):
            return reward_min_queue_squared(x, *args)

    else:
        idx = names.index(target)
        fn = funcs[idx]

        def ret(x, *args):
            return fn(x, *args)

    return ret


def reward_max_speed_count(target_velocity, states, *args):
    """Max. Speed and Count

    Params:
    ------
    * state: ilurl.core.StateCollection
        StateCollection containing --
            ilurl.core.SpeedState and
            ilurl.core.CountState

    Returns:
    --------
    * ret: dict<str, float>
        keys: tls_ids, values: rewards

    Reference:
    ----------
    

    """

    if not ('speed' in states.label and 'count' in states.label):
        raise ValueError(
            'Speed and Count not present in StateCollection')
    else:
        speeds_counts = states.split(('speed', 'count'))
     

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


def reward_min_delay(states, *args):
    """Minimizing the delta of queue length squared 

    Reward definition 1: Minimizing the delay

    Params:
    ------
    * state: ilurl.core.DelayState
        captures the delay experiened by phases.

    Returns:
    --------
    * ret: dict<str, float>
        keys: tls_ids, values: rewards

    Reference:
    ----------
    * El-Tantawy, et al. 2014
        "Design for Reinforcement Learning Parameters for Seamless"

    * Lu, Liu, & Dai. 2008
        "Incremental multistep Q-learning for adaptive traffic signal control"

    * Shoufeng et al., 2008
        "Q-Learning for adaptive traffic signal control based on delay"

    * Abdullhai et al. 2003
        "Reinforcement learning for true adaptive traffic signal control."

    * Wiering, 2000
        "Multi-agent reinforcement learning for traffic light control."

    """
    if 'delay' not in states.label:
        raise ValueError('DelayState not present in StateCollection')
    else:
        state = states.state(('delay',))

    ret = {}
    for tls_id, phase_obs in state.items():
        ret[tls_id] = -sum([dly for obs in phase_obs for dly in obs])
    return ret


def reward_min_queue_squared(state, duration):
    """Minimizing the delta of queue length squared 

    Reward definition 3: Minimizing and Balancing Queue Length

    Params:
    ------
    * state: ilurl.core.QueueState
        queue state has the maximum length lane.

    Returns:
    --------
    * ret: dict<str, float>
        keys: tls_ids, values: rewards

    Reference:
    ----------
    * El-Tantawy, et al. 2014
        "Design for Reinforcement Learning Parameters for Seamless"

    * Balaji, et al. 2010
        "Urban traffic signal control using reinforcement learning agent"

    * De Oliveira et al. 2006
        "Reinforcement learning-based control of traffic lights"

    * Camponogara and Kraus, 2003
        "Distributed learning agents in urban traffic control."
    """
    if 'queue' not in states.label:
        raise ValueError('QueueState not present in StateCollection')
    else:
        state = states.state(('queue',))

    global QUEUE
    ret = {}
    for tls_id, _max_q in state.state.items():
        _max_q = np.concatenate(_max_q, axis=0)
        _prev_q = QUEUE.get(tls_id, np.zeros(_max_q.shape))

        ret[tls_id] = _max_q.dot(_max_q) - _prev_q.dot(_prev_q)
        # 2) Store: if this is a decision point.
        if duration == 0.0:
            QUEUE[tls_id] = _max_q
    return ret
