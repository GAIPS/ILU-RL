from ilurl.utils.meta import RewardsMeta

class MaxMeanSpeedReward(object, metaclass=RewardsMeta):

    def __init__(self, state, mdp_params):
        """Creates a reference to the input state"""
        if state.__class__ != 'MeanSpeedState':
            raise ValueError(
                'MeanSpeed reward expects `MeanSpeedState` state')
        else:
            self.state = state
        self.target_velocity = mdp_params.target_velocity

    def calculate(self):
        counts = self.state.get('MeanState')
        speeds = self.state.get('SpeedState')

        if sum(counts) <= 0:
            reward = 0
        else:
            max_cost = \
                np.array([self.target_velocity] * len(speeds))

            reward = \
                -np.maximum(max_cost - speeds, 0).dot(counts)
        return rewards

# TODO: implement Rewards definition 1
class MinDelayReward(object, metaclass=RewardsMeta):
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
class MinDeltaDelay(object, metaclass=RewardsMeta):
    def __init__(self, state):
        """Creates a reference to the input state"""
        if state.__class__ != 'DeltaDelay':
            raise ValueError(
                'MinDeltaDelay reward expects `DeltaDelay` state')

    def calculate(self):
        return -sum(self._state.to_list())

