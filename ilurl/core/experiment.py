"""Contains an experiment class for running simulations.
   2019-06-2019
   ------------
   This file was copied from flow.core.experiment in order to
   add the following features:
   * periodically save the running data: server seems to
   be restarting every 400 steps, the rewards are being changed
   radically after each restart

   * extend outputs to costumized reward functions
   * fix bug of averaging speeds when no cars are on the simulation
   """
import os
from pathlib import Path
import warnings
import datetime
import json
import logging
import time
from threading import Thread

from tqdm import tqdm

import numpy as np

# TODO: Track those anoying warning
warnings.filterwarnings('ignore')

class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a scenario and environment. In order to use
    it to run an scenario and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> exp = Experiment(env)  # for some env and scenario
        >>> exp.run(num_runs=1, num_steps=1000)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, num_steps=1000, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> sim_params = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, num_steps=1000)

    Attributes
    ----------
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self,
                env,
                exp_path,
                train=True,
                log_info=False,
                log_info_interval=20,
                save_agent=False,
                save_agent_interval=100):
        """

        Parameters
        ----------
        env : flow.envs.Env
            the environment object the simulator will run
        exp_path : str
            path to dump experiment results
        train : bool
            whether to train agent
        log_info : bool
            whether to log experiment info into json file throughout training
        log_info_interval : int
            json file log interval (in number of agent-update steps)
        save_agent : bool
            whether to save RL agent parameters throughout training
        save_agent_interval : int
            save RL agent interval (in number of agent-update steps)

        """
        # Guarantees that the enviroment has stopped.
        if not train:
            env.stop = True

        self.env = env
        self.train = train
        self.exp_path = Path(exp_path) if exp_path is not None else None
        # fails gracefully if an environment with no cycle time
        # is provided
        self.cycle = getattr(env, 'cycle_time', None)
        self.save_step = getattr(env, 'cycle_time', 1)
        self.log_info = log_info
        self.log_info_interval = log_info_interval
        self.save_agent = save_agent
        self.save_agent_interval = save_agent_interval

        logging.info(" Starting experiment {} at {}".format(
            env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")


    def run(
            self,
            num_steps,
            rl_actions=None,
            stop_on_teleports=False
    ):
        """
        Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
        num_steps : int
            number of steps to be performs in each run of the experiment
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        stop_on_teleport : boolean
            if true will break execution on teleport which occur on:
            * OSM scenarios with faulty connections
            * collisions
            * timeouts a vehicle is unable to move for 
            -time-to-teleport seconds (default 300) which is caused by
            wrong lane, yield or jam

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step (last run)

        References
        ---------


        https://sourceforge.net/p/sumo/mailman/message/33244698/
        http://sumo.sourceforge.net/userdoc/Simulation/Output.html
        http://sumo.sourceforge.net/userdoc/Simulation/Why_Vehicles_are_teleporting.html
        """
        if rl_actions is None:

            def rl_actions(*_):
                return None

        info_dict = {}

        vels = []
        vehs = []
        observation_spaces = []
        rewards = []

        veh_i = []
        vel_i = []

        agent_updates_counter = 0

        # Setup logs folder.
        os.makedirs(self.exp_path / 'logs', exist_ok=True)
        train_log_path = self.exp_path / 'logs' / "train_log.json"

        state = self.env.reset()

        for step in tqdm(range(num_steps)):

            # WARNING: This is not synchronized with agents' cycle time.
            # if step % 86400 == 0 and agent_updates_counter != 0: # 24 hours
            #     self.env.reset()

            state, reward, done, _ = self.env.step(rl_actions(state))

            veh_i.append(len(self.env.k.vehicle.get_ids()))
            vel_i.append(
                np.nanmean(self.env.k.vehicle.get_speed(
                    self.env.k.vehicle.get_ids()
                    )
                )
            )

            if self._is_save_step():

                observation_spaces.append(self.env.get_observation_space().state)

                rewards.append(reward)

                vehs.append(np.nanmean(veh_i).round(4))
                vels.append(np.nanmean(vel_i).round(4))
                veh_i = []
                vel_i = []

                agent_updates_counter += 1

                # Save train log (data is aggregated per traffic signal).
                if self.log_info and \
                    (agent_updates_counter % self.log_info_interval == 0):

                    info_dict["rewards"] = rewards
                    info_dict["velocities"] = vels
                    info_dict["vehicles"] = vehs
                    info_dict["observation_spaces"] = observation_spaces
                    info_dict["actions"] = [a for a in self.env.actions_log.values()]
                    info_dict["states"] = [s for s in self.env.states_log.values()]

                    with train_log_path.open('w') as f:
                        t = Thread(target=json.dump(info_dict, f))
                        t.start()

            if done and stop_on_teleports:
                break

            if self.save_agent and self._is_save_agent_step(agent_updates_counter):
                self.env.mas.save_checkpoint(self.exp_path)

        # Save train log (data is aggregated per traffic signal).
        info_dict["rewards"] = rewards
        info_dict["velocities"] = vels
        info_dict["vehicles"] = vehs
        info_dict["observation_spaces"] = observation_spaces
        info_dict["actions"] = [a for a in self.env.actions_log.values()]
        info_dict["states"] = [s for s in self.env.states_log.values()]

        self.env.mas.terminate()
        self.env.terminate()

        return info_dict

    def _is_save_step(self):
        if self.cycle is not None:
            return self.env.duration == 0.0
        return False

    def _is_save_agent_step(self, counter):
        if self.env.duration == 0.0 and counter % self.save_agent_interval == 0:
            return self.train and hasattr(self.env, 'dump') and self.exp_path
        return False
