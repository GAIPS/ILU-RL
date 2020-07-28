"""Implementation of rewards to be used on features space
    
    TODO: Move from strategy pattern to pure-functional
        implementation
"""
import inspect
from sys import modules

import numpy as np



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
    > ('reward_max_speed_count', reward_min_delay')
    > objs
    > (<function ilurl.rewards.reward_max_speed_count(state, *args)>,
       <function ilurl.rewards.reward_min_delay(state, *args)>)
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
        mdp specify: agent, state, rewards, gamma and learning params

    Returns:
    --------
    * reward: function(state)
        generates a reward function that receives state input

    """
    target = mdp_params.reward
    names, funcs = get_rewards()
    if target not in names:
        raise ValueError(f'build_rewards: {target} not in {names}')

    idx = names.index(target)
    fn = funcs[idx]

    # Rescale rewards.
    def ret(x, *args):
        return rescale_rewards(fn(x, *args), scale_factor=mdp_params.reward_rescale)

    return ret


def reward_max_speed_count(state, *args):
    """Max. Speed and Count

    Params:
    ------
    * state: ilurl.state.State
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
    # 1) Splits speed & count
    speeds_counts = state.feature_map(
        filter_by=('speed', 'count'),
        split=True
    )

    # 2) Iterate wrt agents:
    # Unpacks & performs -<speed, count>.
    ret = {k:-np.dot(*v) for k, v in speeds_counts.items()}

    return ret


def reward_min_delay(state, *args):
    """Minimizing the delay

    Reward definition 1: Minimizing the delay

    Params:
    ------
    * state: ilurl.state.State
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
    delays = state.feature_map(
        filter_by=('delay',)
    )
    ret = {}
    for tls_id, phase_obs in delays.items():
        ret[tls_id] = -sum([dly for obs in phase_obs for dly in obs])
    return ret

def reward_max_delay_reduction(state, *args):
    """Max. the reduction on successive delays

    Reward definition 2: Max. the reduction in delay

    OBS: The article uses total delay, which implies
    that the agents are privy to the delays experiencied
    by vehicles wrt other agents. Here we use delay and
    lagged delay only.

    Params:
    ------
    * state: ilurl.state.State
        captures the delay experiened by phases.

    Returns:
    --------
    * ret: dict<str, float>
        keys: tls_ids, values: rewards

    Reference:
    ----------
    * El-Tantawy, et al. 2014
        "Design for Reinforcement Learning Parameters for Seamless"

    * Arel, I., Liu, C., Urbanik, T., & Kohls, A. G. 2010.
        "Reinforcement learning-based multi-agent system for network traffic signal control"

    """
    def diff(x, y):
        return np.array(x) - np.array(y)

    delay_lagdelay = state.feature_map(
        filter_by=('lag[delay]', 'delay'),
        split=True
    )
    ret = {tls_id: np.sum(diff(*del_ldel)).round(4)
           for tls_id, del_ldel in delay_lagdelay.items()}

    return ret

def reward_min_queue_squared(state):
    """Minimizing the delta of queue length squared

    Reward definition 3: Minimizing and Balancing Queue Length

    Params:
    ------
    * state: ilurl.state.State
        queue features has the maximum length lane.

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
    queue_lagqueue = state.feature_map(
        filter_by=('queue', 'lag[queue]'),
        split=True
    )

    ret = {}
    for tls_id, q_lq in queue_lagqueue.items():
        queue, lagqueue = q_lq

        ret[tls_id] = -(np.dot(queue, queue) - np.dot(lagqueue, lagqueue))
    return ret


def rescale_rewards(rewards, scale_factor):
    return {key: r * scale_factor for (key, r) in rewards.items()}
