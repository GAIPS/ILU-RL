import numpy as np
from ilurl.core.meta import MetaReward


def build_rewards(mdp_params):
    """Builder that defines all rewards
    """
    return MaxSpeedCountReward(mdp_params)
    


class MaxSpeedCountReward(object, metaclass=MetaReward):

    def __init__(self,  mdp_params):
        """Creates a reference to the input state"""
        reward_params = mdp_params.reward.additional_params
        if 'target_velocity' not in reward_params:
            raise ValueError('MaxSpeedCountReward must define target_velocity')
        self.target_velocity = reward_params['target_velocity']

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

# TODO: implement Rewards definition 1
class MinDelayReward(object, metaclass=MetaReward):
    def __init__(self, state):
        """Creates a reference to the input state"""
        if state.__class__ != 'Delay':
            raise ValueError(
                'MinDelay reward expects `Delay` state')
        else:
            self._state = state


    # TODO: make state sumable
    def calculate(self):
        return -sum(self._state.to_list())


# TODO: implement Rewards definition 2
class MinDeltaDelay(object, metaclass=MetaReward):
    def __init__(self, state):
        """Creates a reference to the input state"""
        if state.__class__ != 'DeltaDelay':
            raise ValueError(
                'MinDeltaDelay reward expects `DeltaDelay` state')

    def calculate(self):
        return -sum(self._state.to_list())

