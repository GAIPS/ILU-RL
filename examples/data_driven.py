"""This script loads a template from data"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-08'

import time
import os
# from IPython.core.debugger import Pdb

from flow.controllers import GridRouter
from flow.core.params import (EnvParams, InFlows, InitialConfig, NetParams,
                              SumoCarFollowingParams, SumoParams,
                              TrafficLightParams, VehicleParams)

from flow.envs import TestEnv
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios import Scenario

from ilurl.core.experiment import Experiment

EMISSION_PATH = '/Users/gsavarela/sumo_data/'
HORIZON = 1500
NUM_ITERATIONS = 5
SHORT_CYCLE_TIME = 31
LONG_CYCLE_TIME = 45
SWITCH_TIME = 6
# debugger = Pdb()

# feed SOURCES to InitialConfig
# on edges distribution
EDGES_DISTRIBUTION = ["309265401#0"]
SOURCES = [
    "309265401#0",
    "-238059324#1",
    "96864982#0",
    "309265398#0"
]
SINKS = [
    "-309265401#2",
    "306967025#0",
    "238059324#0",
]

EDGES = ["212788159_0", "247123161_0", "247123161_1", "247123161_3",
         "247123161_14", "247123161_4", "247123161_5", "247123161_6",
         "247123161_15", "247123161_7", "247123161_8", "247123161_10",
         "247123161_16", "247123161_11", "247123161_12", "247123161_13",
         "247123161_17", "247123367_0", "247123367_1", "247123374_0",
         "247123374_1", "247123374_3", "247123374_9", "247123374_4",
         "247123374_5", "247123374_6", "247123374_7", "247123449_0",
         "247123449_2", "247123449_1", "247123464_0", "3928875116_0"]


class IntersectionScenario(Scenario):

    def specify_routes(self, net_params):
        rts = {
            "309265401#0": ["238059328#0", "306967025#0"]
        }
        return rts

    def specify_edge_starts(self):
        sts = [("309265401#0", 77.4)]

        return sts

    # probabilistic initializations
    # def specify_routes(self, net_params):
    #     rts = {
    #         "309265401#0": [(["238059324#0"], 0.5),
    #                         (["238059328#0", "306967025#0"], 0.5)],
    #         "-238059324#1": [(["-309265401#2"], 0.5),
    #                          (["238059328#0", "306967025#0"], 0.5)],
    #         "96864982#0": [(["96864982#1", "392619842", "238059324#0"], 0.5),
    #                        (["96864982#1", "392619842", "238059328#0",
    #                          "306967025#0"], 0.5)],
    #         "309265398#0": [(["-306967025#2"], 0.5),
    #                         (["-238059328#2", "-238059328#2"], 0.5)]
    #     }
    #     return rts

    # def specify_edge_starts(self):
    #     sts = [("309265401#0", 0), ("-238059324#1", 0),
    #            ("96864982#0", 0), ("309265398#0", 0)]
    #     return sts

def get_flow_params(flow_sources, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    flow_sources: list of strings
        ids from the edges in which the vehicles come from
    row_num : int
        number of rows in the grid
    additional_net_params : dict
        network-specific parameters that are unique to the grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    # initial = InitialConfig(edges_distribution=SOURCES,
    #                         spacing='random',
    #                         lanes_distribution=float('inf'),
    #                         shuffle=True)


    # as per tutorial

    initial = InitialConfig(edges_distribution=EDGES_DISTRIBUTION)
    # initial = InitialConfig(edges_distribution=SOURCES,
    #                         spacing='random')
    inflow = InFlows()
    for i in range(len(flow_sources)):
        inflow.add(veh_type='human',
                   edge=flow_sources[i],
                   probability=0.25,
                   depart_lane='free',
                   depart_speed=20)

    net = NetParams(inflows=inflow,
                    template=f'{os.getcwd()}/data/networks/intersection.net.xml',
                    additional_params=additional_net_params)

    return initial, net


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    add_net_params.update({'enter_speed': enter_speed})

    initial = InitialConfig(edges_distribution=EDGES_DISTRIBUTION)
    # initial = InitialConfig(edges_distribution=SOURCES,
    #                         spacing='random',
    #                         additional_params=add_net_params)

    net = NetParams(
        template=f'{os.getcwd()}/data/networks/intersection.net.xml',
        additional_params=add_net_params)

    return initial, net


def network_example(render=None,
                    use_inflows=False,
                    additional_env_params=None,
                    emission_path=None,
                    sim_step=0.1):
    """
    Perform a the simulation on a predefined network

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    use_inflows : bool, optional
        set to True if you would like to run the experiment with inflows of
        vehicles from the edges, and False otherwise

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a grid.
    """
    v_enter = 10
    # tot_cars = 160
    tot_cars = 1
    if render is None:
        sim_params = SumoParams(sim_step=sim_step,
                                render=False,
                                print_warnings=False,
                                emission_path=emission_path)

    else:
        sim_params = SumoParams(sim_step=sim_step,
                                render=render,
                                print_warnings=False,
                                emission_path=emission_path)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
        ),
        num_vehicles=tot_cars)

    env_params = EnvParams(horizon=HORIZON,
                           additional_params=ADDITIONAL_ENV_PARAMS)



    tl_logic = TrafficLightParams(baseline=False)
    # phases = [{
    #     "duration": "31",
    #     "minDur": "8",
    #     "maxDur": "45",
    #     "state": "GrGrGrGrGrGr"
    # }, {
    #     "duration": "6",
    #     "minDur": "3",
    #     "maxDur": "6",
    #     "state": "yryryryryryr"
    # }, {
    #     "duration": "31",
    #     "minDur": "8",
    #     "maxDur": "45",
    #     "state": "rGrGrGrGrGrG"
    # }, {
    #     "duration": "6",
    #     "minDur": "3",
    #     "maxDur": "6",
    #     "state": "ryryryryryry"
    # }]
    # # Junction ids
    # tl_logic.add("247123161", phases=phases, programID=1)
    # tl_logic.add("247123374", phases=phases, programID=1)
    # tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")

    # Define flow
    # lookup ids
    additional_net_params = {
        'template': f'{os.getcwd()}/data/networks/intersection.net.xml',
        "speed_limit": 35
    }

    source_edge_ids = [
        "309265401#0",
        "-306967025#2",
        "96864982#0",
        "309265398#0"
    ]
    sink_edge_ids = {
        "-309265401#2",
        "306967025#0",
        "238059324#0",
    }
    if use_inflows:
        initial_config, net_params = get_flow_params(
            edge_ids,
            additional_net_params=additional_net_params)
    else:
        initial_config, net_params = get_non_flow_params(
            enter_speed=v_enter, add_net_params=additional_net_params)

    # TODO: template should be an input variable
    # assumption project gets run from root
    scenario = IntersectionScenario(
        name="intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)


    # debugger.set_trace()
    env = AccelEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario)

    exp = Experiment(env)

    return Experiment(env)


if __name__ == "__main__":
    exp = network_example(
        render=True
    )

    exp.run(NUM_ITERATIONS, HORIZON)