import numpy as np

from flow.envs.base import Env

from ilurl.envs.elements import build_vehicles
from ilurl.state import State
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

        # TLS programs (discrete action space).
        if mdp_params.action_space == 'discrete':
            self.programs = network.programs

        # Keeps the internal value of sim step.
        self.sim_step = sim_params.sim_step
        assert self.sim_step == 1 # step size must equal 1.

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
        self.observation_space = State(network, mdp_params)

        # Continuous action space signal plans.
        self.signal_plans_continous = {}

        self._reset()

    @property
    def duration(self):
        if self.time_counter == 0:
            return 0.0

        if self._duration_counter != self.time_counter:
            self._duration = \
                round(self._duration + self.sim_step, 2) % self.cycle_time
            self._duration_counter = self.time_counter
        return self._duration

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

    def _update_observation_space(self):
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
        if self._update_counter != self.time_counter:
            self._update_observation_space()

            vehs = {nid: {p: snapshots.get(self.duration, [])
                          for p, snapshots in data.items()}
                    for nid, data in self.incoming.items()}

            self.observation_space.update(self.duration, vehs)
            self._update_counter = self.time_counter

        return self.observation_space

    def get_state(self):
        """
        Return the state of the simulation as perceived by the RL agent.

        Returns:
        -------
        state : array_like
            information on the state of the vehicles, which is provided to the
            agent
        """
        obs = self.get_observation_space().feature_map(
            categorize=self.mdp_params.discretize_state_space,
            flatten=True
        )
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
        if self.duration == 0 or self.time_counter == 1:
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

            if dur == 0 and self.tls_type == 'controlled' and \
                self.mdp_params.action_space == 'continuous':
                # Calculate cycle length allocations for the
                # new cycle (continuous action space).

                # Get current action and respective number of phases.
                current_action = np.array(self._current_rl_action()[tid])
                num_phases = len(current_action)

                # Remove yellow time from cycle length (yellow time = 6 seconds).
                available_time = self.cycle_time - (6.0 * num_phases)

                # Allocate time for each of the phases.
                # By default 20% of the cycle length will be equally distributed for 
                # all the phases in order to ensure a minium green time for all phases.
                # The remainder 80% are allocated by the agent.
                phases_durations = np.around(0.2 * available_time * (1 / num_phases) + \
                                    current_action * 0.8 * available_time, decimals=0)

                # Convert phases allocations into a signal plan.
                counter = 0
                timings = []
                for p in range(num_phases):
                    timings.append(counter + phases_durations[p])
                    timings.append(counter + phases_durations[p] + 6.0)
                    counter += phases_durations[p] + 6.0

                timings[-1] = self.cycle_time
                timings[-2] = self.cycle_time - 6.0

                # Store the signal plan. This variable stores the signal
                # plan to be executed throughout the current cycle.
                self.signal_plans_continous[tid] = timings

            if (dur == 0 and self.step_counter > 1):
                return True

            if static:
                return dur in self.tls_durations[tid]
            else:
                if self.mdp_params.action_space == 'discrete':
                    # Discrete action space: TLS programs.
                    progid = self._current_rl_action()[tid]
                    return dur in self.programs[tid][progid]
                else:
                    # Continuous action space: phases durations.
                    return dur in self.signal_plans_continous[tid]

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
        # self.update_observation_space()

        if self.tls_type != 'actuated':
            if self.duration == 0 or self.time_counter == 1:
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

        # # Update timer.
        # self.duration = \
        #     round(self.duration + self.sim_step, 2) % self.cycle_time



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
        return self.reward(self.get_observation_space())

    def reset(self):
        super(TrafficLightEnv, self).reset()
        self._reset()

    def _reset(self):

        # The 'duration' variable measures the elapsed time since
        # the beggining of the cycle, i.e. it measures (in seconds)
        # for how long the current configuration has been going on.
        self._duration_counter = -1
        self._duration = self.time_counter * self.sim_step

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
        # Controls the number of updates
        self._update_counter = -1

