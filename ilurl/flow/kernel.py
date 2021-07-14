"""Script containing the Flow kernel object for interacting with simulators."""

import warnings
from ilurl.flow.simulation_kernel import CityflowSimulation
from ilurl.flow.network_kernel import CityflowKernelNetwork
from ilurl.flow.vehicle_kernel import CityflowVehicle
from ilurl.flow.traffic_light_kernel import CityflowTrafficLight


class Kernel(object):
    """Kernel for abstract function calling across traffic simulator APIs.

    The kernel contains four different subclasses for distinguishing between
    the various components of a traffic simulator.

    * simulation: controls starting, loading, saving, advancing, and resetting
      a simulation in Flow (see flow/core/kernel/simulation/base.py)
    * network: stores network-specific information (see
      flow/core/kernel/network/base.py)
    * vehicle: stores and regularly updates vehicle-specific information. At
      times, this class is optimized to efficiently collect information from
      the simulator (see flow/core/kernel/vehicle/base.py).
    * traffic_light: stores and regularly updates traffic light-specific
      information (see flow/core/kernel/traffic_light/base.py).

    The above kernel subclasses are designed specifically to support
    simulator-agnostic state information calling. For example, if you would
    like to collect the vehicle speed of a specific vehicle, then simply type:

    >>> k = Kernel(simulator="...")  # a kernel for some simulator type
    >>> veh_id = "..."  # some vehicle ID
    >>> speed = k.vehicle.get_speed(veh_id)

    In addition, these subclasses support sending commands to the simulator via
    its API. For example, in order to assign a specific vehicle a target
    acceleration, type:

    >>> k = Kernel(simulator="...")  # a kernel for some simulator type
    >>> veh_id = "..."  # some vehicle ID
    >>> k.vehicle.apply_acceleration(veh_id)

    These subclasses can be modified and recycled to support various different
    traffic simulators, e.g. SUMO, AIMSUN, TruckSim, etc...
    """

    def __init__(self, simulator, sim_params):
        """Instantiate a Flow kernel object.

        Parameters
        ----------
        simulator : str
            simulator type, must be one of {"traci"}
        sim_params : ilurl.flow.params.SimParams
            simulation-specific parameters

        Raises
        ------
        flow.utils.exceptions.FatalFlowError
            if the specified input simulator is not a valid type
        """
        self.kernel_api = None
        self.simulator = simulator

        if  simulator == 'cityflow':
            self.simulation = CityflowSimulation(self)
            self.network = CityflowKernelNetwork(self, sim_params)
            self.vehicle = CityflowVehicle(self, sim_params)
            self.traffic_light = CityflowTrafficLight(self)
        else:
            raise ValueError('Simulator type "{}" is not valid.'.
                                 format(simulator))

    def pass_api(self, kernel_api):
        """Pass the kernel API to all kernel subclasses."""
        self.kernel_api = kernel_api
        self.simulation.pass_api(kernel_api)
        self.network.pass_api(kernel_api)
        self.vehicle.pass_api(kernel_api)
        if (self.simulator == 'cityflow'):
            self.traffic_light.pass_api(kernel_api, self.network)
        else:
            self.traffic_light.pass_api(kernel_api)

    def update(self, reset):
        """Update the kernel subclasses after a simulation step.

        This is meant to support optimizations in the performance of some
        simulators. For example, this step allows the vehicle subclass in the
        "traci" simulator uses the ``update`` method to collect and store
        subscription information.

        Parameters
        ----------
        reset : bool
            specifies whether the simulator was reset in the last simulation
            step
        """
        self.vehicle.update(reset)
        self.traffic_light.update(reset)
        self.network.update(reset)
        self.simulation.update(reset)

    def close(self):
        """Terminate all components within the simulation and network."""
        self.network.close()
        self.simulation.close()

    @property
    def scenario(self):
        """Return network for this deprecated method."""
        warnings.simplefilter('always', PendingDeprecationWarning)
        warnings.warn(
            "self.k.scenario will be deprecated in a future release. Please "
            "use self.k.network instead.",
            PendingDeprecationWarning
        )
        return self.network
