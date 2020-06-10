import numpy as np

class PowerSchedule(object):
    def __init__(self, power_coef):
        """
        Power schedule (input t): 1 / ((1 + t)**power_coef)
        
        Parameters:
        ----------
        power_coef: float
            power coefficient

        """
        self.power_coef = power_coef

    def value(self, t):
        """See Schedule.value"""
        return 1 / np.power(1 + t, self.power_coef)