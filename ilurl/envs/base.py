"""
    Custom environment class.
    (extends flow.envs.base.Env class)

    flow.envs.base.Env.step() methods' calling order:

        1) apply_rl_actions()
        2) Advance simulator by one step.
        3) get_state()
        4) compute_reward()

    For more info see flow.envs.base.Env class.

"""
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

        # Traffic light system type.
        # ('rl', 'static', 'uniform', 'actuated', 'actuated_delay' or 'random).
        self.tls_type = env_params.additional_params.get('tls_type')

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

        # Object that handles the Multi-Agent RL system logic.
        mdp_params.phases_per_traffic_light = network.phases_per_tls
        mdp_params.num_actions = network.num_signal_plans_per_tls
        self.mas = DecentralizedMAS(mdp_params, exp_path, seed)

        # Reward function.
        self.reward = build_rewards(mdp_params)

        self.actions_log = {}
        self.states_log = {}

        # overrides GYM's observation space
        self.observation_space = State(network, mdp_params)
        import ipdb
        ipdb.set_trace()

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

    def get_observation_space(self):
        """ Query kernel to retrieve vehicles' information
        and send it to ilurl.State object for state computation.

        Returns:
        -------
        observation_space: ilurl.State object

        """
        if self._update_counter != self.time_counter:

            # Query kernel and retrieve vehicles' data.
            vehs = {nid: {p: build_vehicles(nid, data['incoming'], self.k.vehicle)
                        for p, data in self.tls_phases[nid].items()}
                            for nid in self.tls_ids}

            self.observation_space.update(self.duration, vehs)

            self._update_counter = self.time_counter

        return self.observation_space

    def get_state(self):
        """ Return the state of the simulation as perceived by the RL agent(s).

        Returns:
        -------
        state : dict

        """
        obs = self.get_observation_space().feature_map(
            categorize=self.mdp_params.discretize_state_space,
            flatten=True
        )
        return obs

    def _rl_actions(self, state):
        """ Return the selected action(s) given the state of the environment.

        Parameters:
        ----------
        state : dict
            Information on the state of the vehicles, which is provided to the
            agent

        Returns:
        -------
        actions: dict

        """
        return self.mas.act(state) # Delegate to Multi-Agent System.

    def _cl_actions(self):
        """ Executes the control actions.

        Returns:
        -------
        cl_actions: tuple<bool>
            False;  duration<state_k> < duration < duration<state_k+1>
            True;  duration == duration<state_k+1>

        """
        ret = []
        dur = int(self.duration)

        def fn(tid):

            if (dur == 0 or self.time_counter == 1) and self.tls_type == 'rl' and \
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
                # all the phases in order to ensure a minimum green time for all phases.
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

            if self.tls_type == 'static':
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
        """ Specify the actions to be performed.

        Parameters:
        ----------
        rl_actions: dict or None
            actions to be performed

        """
        if self.tls_type != 'actuated':

            if (self.tls_type == 'rl' or self.tls_type == 'random') and \
                (self.duration == 0 or self.time_counter == 1):
                # New cycle.

                # Get the number of the current cycle.
                cycle_number = \
                    int(self.step_counter / self.cycle_time)

                # Get current state.
                state = self.get_state()

                # Select new action.
                if rl_actions is None:
                    rl_action = self._rl_actions(state)
                else:
                    rl_action = rl_actions

                self.actions_log[cycle_number] = rl_action
                self.states_log[cycle_number] = state

                if self.step_counter > 1:
                    # RL-agent update.
                    reward = self.compute_reward(None)
                    prev_state = self.states_log[cycle_number - 1]
                    prev_action = self.actions_log[cycle_number - 1]
                    self.mas.update(prev_state, prev_action, reward, state)

            # Update traffic lights' control signals.
            self._apply_cl_actions(self._cl_actions())

        else:
            if self.duration == 0:
                self.observation_space.reset()


    def _apply_cl_actions(self, cl_actions):
        """ For each TSC shift phase or keep phase.

        Parameters:
        ----------
        cl_actions: list<bool>
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
        """ Returns current action. """
        N = (self.cycle_time / self.sim_step)
        actid = \
            int(max(0, self.step_counter - 1) / N)

        return self.actions_log[actid]

    def compute_reward(self, rl_actions, **kwargs):
        """ Reward calculation.
        
        Parameters:
        ----------
        rl_actions : array_like
            Actions performed by each TSC.

        kwargs : dict
            Other parameters of interest.

        Returns:
        -------
        reward : dict<float>
            Reward at each TSC.

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

        # Stores the state index.
        self.state_indicator = {}
        for node_id in self.tls_ids:
            num_phases = len(self.tls_phases[node_id])
            if self.tls_type != 'actuated':
                self.state_indicator[node_id] = 0
                s0 = self.tls_states[node_id][0]
                self.k.traffic_light.set_state(node_id=node_id, state=s0)

        # Observation space.
        self.observation_space.reset()

        # Controls the number of updates.
        self._update_counter = -1
