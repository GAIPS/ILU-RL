import abc

class MASInterface(abc.ABC):
    """
        Multi-agent system interface.
    """

    @abc.abstractmethod
    def act(self, s):
        """ Act. """

    @abc.abstractmethod
    def update(self, s, a, r, s1): 
        """ Update. """

    @abc.abstractmethod
    def save_checkpoint(self, path):
        """ Save checkpoint. """

    @abc.abstractmethod
    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        """ Load checkpoint. """

    @abc.abstractmethod
    def terminate(self):
        """ Terminate processes. """