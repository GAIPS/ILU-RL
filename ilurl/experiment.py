"""
    Experiment class to run simulations.
    (See flow.core.experiment)
"""
import os
from pathlib import Path
import warnings
import datetime
import logging

from tqdm import tqdm

import numpy as np

from ilurl.utils.precision import action_to_double_precision

# TODO: Track those anoying warning
warnings.filterwarnings('ignore')

class Experiment:
    """
    Class to run an experiment in any supported simulator.

    This class acts as a runner for a scenario and environment:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> exp = Experiment(env, ...)  # for some env and scenario
        >>> exp.run(num_steps=1000)

    If you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> sim_params = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object.

    """
    def __init__(self,
                env,
                exp_path : str,
                train : bool = True,
                save_agent : bool = False,
                save_agent_interval : int = 100,
                tls_type : str = 'rl'):
        """

        Parameters:
        ----------
        env : flow.envs.Env
            the environment object the simulator will run.
        exp_path : str
            path to dump experiment-related objects.
        train : bool
            Whether to train RL agents.
        save_agent : bool
            Whether to save RL agents parameters throughout training.
        save_agent_interval : int
            RL agent save interval (in number of agent-update steps).
        tls_type : str
            Traffic ligh system type: ('rl', 'random', 'actuated',
            'static', 'webster' or 'max_pressure')

        """
        # Guarantees that the enviroment has stopped.
        if not train:
            env.stop = True

        self.env = env
        self.train = train
        self.exp_path = Path(exp_path) if exp_path is not None else None
        self.cycle = getattr(env, 'cycle_time', None)
        self.save_agent = save_agent
        self.save_agent_interval = save_agent_interval
        self.tls_type = tls_type

        logging.info(" Starting experiment {} at {}".format(
            env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(
            self,
            num_steps : int,
            rl_actions = None,
            stop_on_teleports : bool = False,
            communicate : bool = True
    ):
        """
        Run the given scenario (env) for a given number of steps.

        Parameters:
        ----------
        num_steps : int
            number of steps to run.
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any).
        stop_on_teleport : boolean
            if true will break execution on teleport which occur on:
            * OSM scenarios with faulty connections.
            * collisions.
            * timeouts a vehicle is unable to move for 
            -time-to-teleport seconds (default 300) which is caused by
            wrong lane, yield or jam.

        Returns:
        -------
        info_dict : dict
            contains experiment-related data.

        References:
        ---------
        https://sourceforge.net/p/sumo/mailman/message/33244698/
        http://sumo.sourceforge.net/userdoc/Simulation/Output.html
        http://sumo.sourceforge.net/userdoc/Simulation/Why_Vehicles_are_teleporting.html

        """
        if rl_actions is None:

            def rl_actions(*_):
                return None

        info_dict = {}

        observation_spaces = []
        rewards = []
        vels = []
        vehs = []

        veh_ids = []
        veh_speeds = []

        agent_updates_counter = 0

        state = self.env.reset()


        if communicate:
            alpha = get_alpha(self.env.network.network_id)

        for step in tqdm(range(num_steps)):

            # WARNING: Env reset is not synchronized with agents' cycle time.
            # if step % 86400 == 0 and agent_updates_counter != 0: # 24 hours
            #     self.env.reset()

            state, reward, done, _ = self.env.step(rl_actions(state))
            vehicles = self.env.k.vehicle

            step_ids = vehicles.get_ids()
            step_speeds = [s for s in vehicles.get_speed(step_ids) if s > 0]

            veh_ids.append(len(step_ids))
            veh_speeds.append(np.nanmean(step_speeds))

            if communicate and step > 10:
                import ipdb; ipdb.set_trace()
                # READ: each agent reads their speed.
                x1, y1 = get_x("grid", "247123161",  vehicles), get_y("grid", "247123161",  vehicles)
                x2, y2 = get_x("grid", "247123464",  vehicles), get_y("grid", "247123464",  vehicles)
                x3, y3 = get_x("grid", "247123468",  vehicles), get_y("grid", "247123468",  vehicles)

                vs = np.array([x1, x2, x3])
                vbar = np.array([np.nanmean(step_speeds)] * 3)

                # Consensus Loop: each agent broadcasts and updates
                num_com = 0
                while not np.allclose(vs, vbar):
                    x1, x2, x3 = x1 + alpha * 1 * (x2 - x1), \
                                 x2 + alpha * 2 * (0.5 *  (x1 + x3) - x2), \
                                 x3 + alpha * 1 * (x2 - x3)

                    y1, y2, y3 = y1 + alpha * 1 * (y2 - y1), \
                                 y2 + alpha * 2 * (0.5 *  (y1 + y3) - y2), \
                                 y3 + alpha * 1 * (y2  - y3)

                    vs = np.nan_to_num(np.array([x1 / y1, x2 / y2, x3 / y3]))
                    num_com += 1
                print(step, veh_speeds[-1], num_com)
            else:
                print(step, veh_speeds[-1])

            if self._is_save_step():

                observation_spaces.append(
                    self.env.observation_space.feature_map())

                rewards.append(reward)

                vehs.append(np.nanmean(veh_ids).round(4))
                vels.append(np.nanmean(veh_speeds).round(4))
                veh_ids = []
                veh_speeds = []

                agent_updates_counter += 1

            if done and stop_on_teleports:
                break

            if self.save_agent and self.tls_type == 'rl' and \
                self._is_save_agent_step(agent_updates_counter):
                self.env.tsc.save_checkpoint(self.exp_path)

        # Save train log (data is aggregated per traffic signal).
        info_dict["rewards"] = rewards
        info_dict["velocities"] = vels
        info_dict["vehicles"] = vehs
        info_dict["observation_spaces"] = observation_spaces
        info_dict["actions"] = action_to_double_precision([a for a in self.env.actions_log.values()])
        info_dict["states"] = [s for s in self.env.states_log.values()]

        if self.tls_type not in ('static', 'actuated'):
            self.env.tsc.terminate()
        self.env.terminate()

        return info_dict

    def _is_save_step(self):
        if self.cycle is not None:
            return self.env.duration == 0.0
        return False

    def _is_save_agent_step(self, counter):
        if self.env.duration == 0.0 and counter % self.save_agent_interval == 0:
            return self.train and self.exp_path
        return False


def get_alpha(network_id):
    """
    Communication matrix
    """
    assert network_id == "grid", "W is only defined for grid"

    A = np.array([[0, 1, 0], [1 , 0 , 1], [0, 1, 0]])

    D = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])

    L = D - A

    # eigvals are decreasing 
    eigvals = np.linalg.eigvals(L)

    alpha = 2/ (np.round(eigvals[0], 0) + np.round(eigvals[-2], 0))

    return alpha


def get_x(network_id, tls_id, vehicles):
    """
    Gets speed for a given agent tls_id
    """
    assert network_id == "grid", "W is only defined for grid"

    network = {"247123161": [":247123161_0", ":247123161_1", ":247123161_2", ":247123161_3",
                             ":247123161_4", ":247123161_4", ":24712", "3161_6", ":247123161_7",
                             ":247123161_8", ":247123161_9", ":247123161_10", ":247123161_10",
                             ":247123161", "383432312", "-383432312", "309265401", "-238059324",
                             "238059324", "-238059328", "238059328"],
               "247123464": [":247123464_0", ":247123464_1", ":247123464_1", ":247123464_3",
                             ":247123464_4", ":247123464_5", ":247123464_6", ":247123464_7",
                             ":247123464_8", ":247123464_8", ":247123464", "3092655395#1",
                             "309265400", "-309265401", "22941893"],
               "247123468": [":247123468_0", ":247123468_1", ":247123468_2", ":247123468_3",
                             ":247123468_4", ":247123468_5", ":247123468_5", ":247123468_7",
                             ":247123468_8", ":247123468_8", ":247123468", "23148196",  "309265402",
                             "-309265402", "309265396#1", "-309265400"]}
    edges = network[tls_id]
    speeds = vehicles.get_speed(vehicles.get_ids_by_edge(edges))
    return np.nan_to_num(np.nansum(speeds))

def get_y(network_id, tls_id, vehicles):
    """
    Gets speed for a given agent tls_id
    """
    network = {"247123161": [":247123161_0", ":247123161_1", ":247123161_2", ":247123161_3",
                             ":247123161_4", ":247123161_4", ":24712", "3161_6", ":247123161_7",
                             ":247123161_8", ":247123161_9", ":247123161_10", ":247123161_10",
                             ":247123161", "383432312", "-383432312", "309265401",
                             "-238059324", "238059324", "238059328", "-238059328"],
               "247123464": [":247123464_0", ":247123464_1", ":247123464_1", ":247123464_3",
                             ":247123464_4", ":247123464_5", ":247123464_6", ":247123464_7",
                             ":247123464_8", ":247123464_8", ":247123464", "3092655395#1",
                             "309265400", "-309265401", "22941893"],
               "247123468": [":247123468_0", ":247123468_1", ":247123468_2", ":247123468_3",
                             ":247123468_4", ":247123468_5", ":247123468_5", ":247123468_7",
                             ":247123468_8", ":247123468_8", ":247123468", "23148196",  "309265402",
                             "-309265402", "309265396#1", "-309265400"]}
    edges = network[tls_id]
    vehids = vehicles.get_ids_by_edge(edges)
    return len(vehids)
