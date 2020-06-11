"""Metaclass module to help enforce constraints on derived classes"""
__author__ = 'Guilherme Varela'
__date__ = '2020-03-25'

class MetaAgent(type):
    """Agent: type classes must implement these methods.

    Methods:
    -------

    * act: Specify the actions to be performed by the RL agent(s).

        Parameters:
        -----------
        * s: tuple
            state representation.

        Returns:
        -------
        * a: int
            selected action


    * update: Performs a learning update step.

        Parameters:
        -----------
        * s: tuple 
            state representation.

        * a: int
            action.

        * r: float
            reward.

        * s1: tuple
            state representation.
    

    * stop: Stops agent's learning.

        Parameters:
        ----------
        * stop: bool
            if True agent's learning process is stopped.


    * save_checkpoint: Saves agent's weights into the
                        provided folder path.

        Parameters:
        ----------
        * path: str
            path to save directory.


    *load_checkpoint: Loads model's weights from file.
 
        Parameters:
        ----------
        * chkpts_dir_path: str
            path to checkpoint's directory.

        * chkpt_num: int
            the number of the checkpoint to load.


    * setup_logger: Setup train logger (tensorboard).

        Parameters:
        ----------
        * path: str
            path to log directory.

    References:
    ----------

    https://docs.python.org/3/reference/datamodel.html#metaclasses
    https://realpython.com/python-metaclasses/

    """
    def __new__(meta, name, base, body):
        agent_methods = ('act',
                         'update',
                         'stop',
                         'save_checkpoint',
                         'load_checkpoint',
                         'setup_logger')
        for attr in agent_methods:
            if attr not in body:
                raise TypeError(f'AgentQ must implement {attr}')

        return super().__new__(meta, name, base, body)


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


