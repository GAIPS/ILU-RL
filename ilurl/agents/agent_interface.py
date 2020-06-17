import abc

class AgentInterface(abc.ABC):
    """
        Defines the interface for a single agent.
    """

    @abc.abstractmethod
    def init(self, params, exp_path, name):
        """ Instantiate agent.

            Parameters:
            ----------
            * params: object agent's parameters.

            * exp_path: str
                Path to experiment's directory.

            * name: str

        """

    @abc.abstractmethod
    def act(self, s):
        """ Specify the actions to be performed by the RL agent.

            Parameters:
            ----------
            * s: tuple
                state representation.

        """

    @abc.abstractmethod
    def update(self, s, a, r, s1): 
        """ Performs a learning update step.

            Parameters:
            ----------
            * a: int
                action.

            * r: float
                reward.

            * s1: tuple
                state representation.

        """

    @abc.abstractmethod
    def save_checkpoint(self, path):
        """ Save models' weights.

            Parameters:
            ----------
            * path: str 
                path to save directory.
        """

    @abc.abstractmethod
    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        """ Loads models' weights from files.
 
            Parameters:
            ----------
            * chkpts_dir_path: str
                path to checkpoints' directory.

            * chkpt_num: int
                the number of the checkpoints to load.
        """
