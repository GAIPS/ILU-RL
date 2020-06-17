"""Metaclass module to help enforce constraints on derived classes"""
__author__ = 'Guilherme Varela'
__date__ = '2020-03-25'


class MetaState(type):
    """Establishes common interface for observed states

    References:
    ----------

    https://docs.python.org/3/reference/datamodel.html#metaclasses
    https://realpython.com/python-metaclasses/

    """
    def __new__(meta, name, base, body):

        state_methods = ('tls_phases',
                         'tls_ids',
                         'reset',
                         'update',
                         'label',
                         'state')

        for attr in state_methods:
            if attr not in body:
                raise TypeError(f'State must implement {attr}')

        return super().__new__(meta, name, base, body)


class MetaStateCollection(MetaState):

    def __new__(meta, name, base, body):

        state_collection_methods = ('split',)

        for attr in state_collection_methods:
            if attr not in body:
                raise TypeError(f'State must implement {attr}')

        return super().__new__(meta, name, base, body)


class MetaStateCategorizer(type):
    """Adaptive Traffic Signal Control (ATSC): 
        is a domain that demands function approximations

    """
    def __new__(meta, name, base, body):

        categorizer_methods = ('categorize',)

        for attr in categorizer_methods:
            if attr not in body:
                raise TypeError(f'State must implement {attr}')

        return super().__new__(meta, name, base, body)


class MetaReward(type):
    """Common methods all reward-type objects must implement


    References:
    ----------

    https://docs.python.org/3/reference/datamodel.html#metaclasses
    https://realpython.com/python-metaclasses/
    """

    def __new__(meta, name, base, body):

        reward_methods = ('calculate',)
        for attr in reward_methods:
            if attr not in body:
                raise TypeError(f'Rewards must implement {attr}')

        return super().__new__(meta, name, base, body)

