"""Script containing the base simulation kernel class."""
from cityflow import Engine

# Number of retries on restarting SUMO before giving up

class KernelSimulation(object):
    """Base simulation kernel.

    The simulation kernel is responsible for generating the simulation and
    passing to all other kernel the API that they can use to interact with the
    simulation.

    The simulation kernel is also responsible for advancing, resetting, and
    storing whatever simulation data is relevant.

    All methods in this class are abstract and must be overwritten by other
    child classes.
    """

    def __init__(self, master_kernel):
        """Initialize the simulation kernel.

        Parameters
        ----------
        master_kernel : flow.kernel.Kernel
            the higher level kernel (used to call methods from other
            sub-kernels)
        """
        self.master_kernel = master_kernel
        self.kernel_api = None

    def pass_api(self, kernel_api):
        """Acquire the kernel api that was generated by the simulation kernel.

        Parameters
        ----------
        kernel_api : any
            an API that may be used to interact with the simulator
        """
        self.kernel_api = kernel_api

    def start_simulation(self, network, sim_params):
        """Start a simulation instance.

        network : any
            an object or variable that is meant to symbolize the network that
            is used during the simulation. For example, in the case of sumo
            simulations, this is (string) the path to the .sumo.cfg file.
        sim_params : ilurl.flow.params.SimParams
            simulation-specific parameters
        """
        raise NotImplementedError

    def simulation_step(self):
        """Advance the simulation by one step.

        This is done in most cases by calling a relevant simulator API method.
        """
        raise NotImplementedError

    def update(self, reset):
        """Update the internal attributes of the simulation kernel.

        Any update operations are meant to support ease of simulation in
        current and future steps.

        Parameters
        ----------
        reset : bool
            specifies whether the simulator was reset in the last simulation
            step
        """
        raise NotImplementedError

    def check_collision(self):
        """Determine if a collision occurred in the last time step.

        Returns
        -------
        bool
            True if collision occurred, False otherwise
        """
        raise NotImplementedError

    def close(self):
        """Close the current simulation instance."""
        raise NotImplementedError

"""Script containing the CITY simulation kernel class."""

RETRIES_ON_ERROR = 10


class CityflowSimulation(KernelSimulation):
    """Sumo simulation kernel.

    Extends flow.simulation.KernelSimulation
    """

    def __init__(self, master_kernel):
        """Instantiate the sumo simulator kernel.

        Parameters
        ----------
        master_kernel : flow.kernel.Kernel
            the higher level kernel (used to call methods from other
            sub-kernels)
        """
        super(CityflowSimulation, self).__init__(master_kernel)

    def pass_api(self, kernel_api):
        """See parent class.

        Also initializes subscriptions.
        """
        super(CityflowSimulation, self).pass_api(kernel_api)

    def simulation_step(self):
        """See parent class.
            Advances simulation one step forward
        """
        self.kernel_api.next_step()

    def update(self, reset):
        """See parent class."""
        pass

    def close(self):
        """CityFlow never closes!"""
        pass

    def check_collision(self):
        """See parent class."""
        # CityFlow doesn't check for collisions
        return False

    def start_simulation(self, network, sim_params):
        """Start a CityFlow simulation instance.

        This method uses the configuration files created by the network class
        to initialize a CityFlow instance.

        Should create the logfiles
        """
        seed = 0
        if (sim_params.seed is not None): seed = sim_params.seed

        self.eng = Engine(network.cfg_rel_path.as_posix(), thread_num=8)
        self.eng.set_random_seed(seed)
        self.eng.reset(seed=False)  # Do not reset random seed
        return self.eng
