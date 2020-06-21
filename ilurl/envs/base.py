'''
    Traffic Light Environment.
'''
__date__ = "2019-12-10"
import numpy as np

from flow.core import rewards
from flow.envs.base import Env

from ilurl.envs.elements import build_vehicles
from ilurl.states import build_states
from ilurl.rewards import build_rewards

from ilurl.utils.properties import lazy_property

# TODO: make this a factory in the future.
from ilurl.mas.decentralized import DecentralizedMAS


class TrafficLightEnv(Env):
    """
        Environment used to train traffic light systems.
    """
    def __init__(self,
                 env_params,
                 sim_params,
                 mdp_params,
                 network,
                 exp_path,
                 seed,
                 simulator='traci'):

        super(TrafficLightEnv, self).__init__(env_params,
                                              sim_params,
                                              network,
                                              simulator=simulator)


        # TODO: Allow for mixed networks with actuated,
        # controlled and static traffic light configurations.
        self.tls_type = env_params.additional_params.get('tls_type')

        # Whether TLS timings are static or controlled by agent.
        self.static = (self.tls_type == 'static')

        # Cycle time.
        self.cycle_time = network.cycle_time

        # TLS programs.
        self.programs = network.programs

        # Keeps the internal value of sim step.
        self.sim_step = sim_params.sim_step

        # Problem formulation params.
        self.mdp_params = mdp_params

        # Object that handles the Multi-Agent RL System logic.
        mdp_params.phases_per_traffic_light = network.phases_per_tls
        mdp_params.num_actions = network.num_signal_plans_per_tls
        self.mas = DecentralizedMAS(mdp_params, exp_path, seed)

        # Reward function.
        self.reward = build_rewards(mdp_params)

        self.actions_log = {}
        self.states_log = {}

        # overrides GYM's observation space
        self.observation_space = build_states(network, mdp_params)

        self._reset()

    # overrides GYM's observation space
    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    # TODO: create a delegate class
    @property
    def stop(self):
        return self.mas.stop

    @stop.setter
    def stop(self, stop):
        self.mas.stop = stop

    # TODO: restrict delegate property to an
    # instance of class
    @lazy_property
    def tls_ids(self):
        return self.network.tls_ids

    @lazy_property
    def tls_phases(self):
        return self.network.tls_phases

    @lazy_property
    def tls_states(self):
        return self.network.tls_states

    @lazy_property
    def tls_durations(self):
        return {
            tid: np.cumsum(durations).tolist()
            for tid, durations in self.network.tls_durations.items()
        }

    def update_observation_space(self):
        """
        Updates the observation space.

        Assumes that each traffic light carries a speed sensor.
        (counts and speeds)

        Updates the following data structures:

        * incoming: nested dict
            1st order (outer) keys: int
                    traffic_light_id

            2nd order keys: int
                    TLS phase

            3rd order (inner) keys: float
                    frame_id of observations ranging from 0 to duration 

            values: list
                    vehicle ids and speeds for the given TLS, phase and edges

            """

        for node_id in self.tls_ids:
            for phase, data in self.tls_phases[node_id].items():
                vehs = build_vehicles(node_id, data['components'],
                                      self.k.vehicle)
                self.incoming[node_id][phase][self.duration] = vehs

    def get_observation_space(self):
        """
        Consolidates the observation space.
        Aggregates all data belonging to a complete cycle.

        Update:
        ------
        observation space is now a 3 level hierarchial dict:

            *   intersection: dict
                the top most represents the traffic lights
                (traffic_light_id)

            *   phases: dict
                the second layer represents the phases components
                for each intersection/traffic light

            *   values: list
                the third and final layer represents the variables
                being observed by the agent

        WARNING:
            when all cars are dispatched the
            state will be encoded with speed zero --
            change for when there aren't any cars
            the state is equivalent to maximum speed
        """

        def delay(t):
            return round(
                t - self.sim_step
                if t >= self.sim_step else
                self.cycle_time - self.sim_step
                if self.step_counter > 1 else 0.0, 2)
        prev = delay(self.duration)

        vehs = {nid: {p: snapshots.get(prev, [])
                      for p, snapshots in data.items()}
                for nid, data in self.incoming.items()}

        self.observation_space.update(prev, vehs)

        return self.observation_space

    def get_state(self):
        """
        Return the state of the simulation as perceived by the RL agent.
        
        Returns
        -------
        state : array_like
            information on the state of the vehicles, which is provided to the
            agent
        """
        obs = self.get_observation_space().state

        # Categorize.
        if self.mdp_params.discretize_state_space:
            obs = self.get_observation_space().categorize()

        obs = self.get_observation_space().flatten(obs)

        return obs

    def rl_actions(self, state):
        """
        Return the selected action given the state of the environment.

        Params:
        ------
            state : dict
            information on the state of the vehicles, which is provided to the
            agent
        
        Returns:
        -------
            action : array_like
                information on the state of the vehicles, which is
                provided to the agent

        """
        if self.duration == 0:
            action = self.mas.act(state)
        else:
            action = None

        return action

    def cl_actions(self, static=False):
        """Executes the control action according to a program
            
        Params:
        ------
            * static: boolean
                If true execute the default program or change states at
                duration == tls_durations for each tls.
                Otherwise; (i) fetch the rl_action, (ii) fetch program,
                (iii) execute control action for program
        Returns:
        -------
            * cl_actions: tuple<bool>
                False;  duration<state_k> < duration < duration<state_k+1>
                True;  duration == duration<state_k+1>
        """
        ret = []
        dur = int(self.duration)

        def fn(tid):
            if (dur == 0 and self.step_counter > 1):
                return True

            if static:
                return dur in self.tls_durations[tid]
            else:
                progid = self._current_rl_action()[tid]
                return dur in self.programs[tid][progid]

        ret = [fn(tid) for tid in self.tls_ids]

        return tuple(ret)

    def apply_rl_actions(self, rl_actions):
        """
        Specify the actions to be performed by the RL agent(s).

        Parameters:
        ----------
        rl_actions: list of actions or None
        """

        # Update observation space.
        self.update_observation_space()

        if self.tls_type != 'actuated':
            if self.duration == 0:
                # New cycle.

                # Get the number of the current cycle.
                cycle_number = \
                    int(self.step_counter / self.cycle_time)

                # Get current state.
                state = self.get_state()

                # Select new action.
                if rl_actions is None:
                    rl_action = self.rl_actions(state)
                else:
                    rl_action = rl_actions

                self.actions_log[cycle_number] = rl_action
                self.states_log[cycle_number] = state

                if self.step_counter > 1: # and not self.stop:
                    # RL-agent update.
                    reward = self.compute_reward(None)
                    prev_state = self.states_log[cycle_number - 1]
                    prev_action = self.actions_log[cycle_number - 1]
                    self.mas.update(prev_state, prev_action, reward, state)

            # Update traffic lights' control signals.
            self._apply_cl_actions(self.cl_actions(static=self.static))
        else:
            if self.duration == 0:
                self.observation_space.reset()
                

        # Update timer.
        self.duration = \
            round(self.duration + self.sim_step, 2) % self.cycle_time

    def _apply_cl_actions(self, cl_actions):
        """For each tls shift phase or keep phase

        Params:
        -------
            * cl_actions: list<bool>
                False; keep state
                True; switch to next state
        """
        for i, tid in enumerate(self.tls_ids):
            if cl_actions[i]:
                states = self.tls_states[tid]
                self.state_indicator[tid] = \
                        (self.state_indicator[tid] + 1) % len(states)
                next_state = states[self.state_indicator[tid]]
                self.k.traffic_light.set_state(
                    node_id=tid, state=next_state)
            
    def _current_rl_action(self):
        """Returns current rl action"""
        # adjust for duration
        N = (self.cycle_time / self.sim_step)
        actid = \
            int(max(0, self.step_counter - 1) / N)
        return self.actions_log[actid]

    def compute_reward(self, rl_actions, **kwargs):
        """
        Reward function for the RL agent(s).
        Defaults to 0 for non-implemented environments.
        
        Parameters
        ----------
        rl_actions : array_like
            actions performed by rl vehicles or None

        kwargs : dict
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise

        Returns
        -------
        reward : float or list of float
        """
        return self.reward(self.get_observation_space(), self.duration)

    def reset(self):
        super(TrafficLightEnv, self).reset()
        self._reset()

    def _reset(self):

        # duration measures the amount of time the current
        # configuration has been going on
        self.duration = 0.0

        self.incoming = {}

        # stores the state index
        # used for opt iterations that did not us this variable
        self.state_indicator = {}
        for node_id in self.tls_ids:
            num_phases = len(self.tls_phases[node_id])
            if self.tls_type != 'actuated':
                self.state_indicator[node_id] = 0
                s0 = self.tls_states[node_id][0]
                self.k.traffic_light.set_state(node_id=node_id, state=s0)

            self.incoming[node_id] = {p: {} for p in range(num_phases)}


        self.observation_space.reset()

