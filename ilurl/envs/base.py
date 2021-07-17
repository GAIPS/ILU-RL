"""
    Custom environment class.
    (extends flow.envs.base.Env class)

    flow.envs.base.Env.step() methods' calling order:

        1) Increment counters:
            self.time_counter += 1
            self.step_counter += 1
        2) apply_rl_actions()
        3) Advance simulator by one step.
        4) get_state()
        5) compute_reward()

    For more info see flow.envs.base.Env class.

"""
from operator import itemgetter
from collections import defaultdict
import numpy as np

from ilurl.flow.envs.base import Env

from ilurl.envs.elements import build_vehicles
from ilurl.state.state import State
from ilurl.rewards import build_rewards
from ilurl.utils.properties import lazy_property

from ilurl.envs.controllers import get_ts_controller, is_controller_periodic

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
        # ('rl', 'random', 'static', 'webster', 'actuated' or 'max-pressure').
        self.ts_type = env_params.additional_params.get('tls_type')

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
        mdp_params.phases_per_traffic_light = network.phases_per_tls
        mdp_params.num_actions = network.num_signal_plans_per_tls

        if self.ts_type in ('rl', 'random'):
            self.tsc = DecentralizedMAS(mdp_params, exp_path, seed)
        elif self.ts_type in ('max_pressure', 'webster'):
            self.tsc = get_ts_controller(self.ts_type, network.phases_per_tls,
                                        self.tls_phases, self.cycle_time)
        elif self.ts_type in ('static', 'actuated'):
            pass # Nothing to do here.
        else:
            raise ValueError(f'Unknown ts_type:{self.ts_type}')

        # Reward function.
        self.reward = build_rewards(mdp_params)

        # Overrides GYM's observation space.
        # self.observation_space = State(network, mdp_params)
        self.actions_log = {}
        self.states_log = {}

        # state WAVE
        # |state|  = |p| + 2
        self._cached_state = {tlid: tuple([0] * (np + 2)) for tlid, np in network.phases_per_tls.items()}
        self._duration = defaultdict(lambda : 0)
        self._duration_counter = -1

        # Continuous action space signal plans.
        self.signal_plans_continous = {}

        self._reset()

    #ABCMeta
    def action_space(self):
        return self.mdp_params.action_space

    @property
    def duration(self):
        return self._duration

    # @property
    # def active_duration(self):
    #     # Counts the time since last switch has taken place.
    #     if not any(self.active_phases):
    #         self._active_duration = defaultdict(lambda : 0)
    #     return self._active_duration

    @property
    def active_phases(self):
        # Defines the phase
        return self._active_phases

    def update_active_phases(self, active_phases):
        """ Phases which are active on this timestep
            Where either:
            1) Were active previous time step. 
            2) Have been active as a consequence of
            decision making by the agent and NOT low
            level traffic signal control.
        """
        # Prevents function to be called twice on the same time step.
        if self._duration_counter != self.time_counter:
            assert self._active_phases.keys() == active_phases.keys()

            # "Aliasing": keep notation shorter
            prev = self._active_phases
            this = active_phases
            dur = self._duration

            # this == prev -> increment duration
            for tid, p in this.items():
                dur[tid] = (dur[tid] + 1) if p == prev[tid] else 0
                prev[tid] = p

            # Prevents being updated twice
            self._duration_counter = self.time_counter
        return self._duration

    # overrides GYM's observation space
    # @property
    # def observation_space(self):
    #     return self._observation_space

    # @observation_space.setter
    # def observation_space(self, observation_space):
    #     self._observation_space = observation_space


    # TODO: create a delegate class
    @property
    def stop(self):
        return self.tsc.stop

    @stop.setter
    def stop(self, stop):
        self.tsc.stop = stop

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
        # Timings used by 'static' ts_type.
        return {
            tid: np.cumsum(durations).tolist()
            for tid, durations in self.network.tls_durations.items()
        }


    def update_observation_space(self):
        """ Query kernel to retrieve vehicles' information
        and send it to ilurl.State object for state computation.

        TODO: Choose Phase transform in get
        Returns:
        -------
        observation_space: ilurl.State object

        """
        print(f'{self.time_counter}:update_observation_space')
        # Query kernel and retrieve vehicles' data.
        vehs = {nid: {p: build_vehicles(nid, data['incoming'], self.k.vehicle)
                    for p, data in self.tls_phases[nid].items()}
                        for nid in self.tls_ids}

         
        # TODO: Optimize this computation by using CityFlowSimulatorKernel
        new_vehs = {
            nid: {p: len(phase) for p, phase in phases.items()}
            for nid, phases  in vehs.items()
        }
        state = {}
        self._this_state = {}
        for tlid, phases_dict  in new_vehs.items():
            # order by phase(num) and get values (counts).
            _, wave = zip(*sorted(phases_dict.items(), key=itemgetter(0)))
            state[tlid] = (self.active_phases[tlid], self.duration[tlid]) + wave 
        self._cached_state = state
        # self.observation_space.update(self.duration, vehs)

    def get_state(self):
        """ Return the state of the simulation as perceived by the RL agent(s).

        Returns:
        -------
        state : dict

        """
        print(f'{self.time_counter}:get_state()')
        # TODO: Choose Phase
        # Make value function approximation here
        # min_green and max_green contraint
        # if self.mdp_params.discretize_state_space:
        #    def fn(dur):
        #        if dur < self.min_green: return 0 
        #        if dur < int(self.max_green / 2): return 1
        #        if dur < self.max_green: return 2 
        #        return 3 
        # active_state = {
        #         tlid: [fn(value) if itr == 2 else value
        #         for itr, value in enumerate(values)]
        #     for tlid, values in self._cached_state.items()
        # }
        # k, v
        return self._cached_state
        # obs = self.observation_space.feature_map(
        #     categorize=self.mdp_params.discretize_state_space,
        #     flatten=True
        # )
        # return obs


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

        # Choose Phases: add some tensorflow sheneningans
        # tf_state = {k: np.array(values).astype(np.float32).reshape((1, len(values)))
        #           for k, values in state.items()}
        # if self.time_counter >= 4:
        #     import ipdb; ipdb.set_trace()

        # TUPLE
        tf_state = {tlid: tuple([float(s) for s in sta]) for tlid, sta in state.items()}
        # tf_state = {tlid: np.array([s for s in sta]).astype(np.float32).reshape((1, len(sta))) for tlid, sta in state.items()}
        # print(tf_state['247123161'], tf_state['247123161'].shape, tf_state['247123161'].dtype)
        # Delegate to Multi-Agent System. 
        this_actions = self.tsc.act(state) 

        return this_actions

    def _periodic_control_actions(self):
        """ Executes pre-timed periodic control actions.

        * Actions generate a plan for one period ahead e.g cycle time.

        Returns:
        -------
        controller_actions: tuple<bool>
            False;  duration<state_k> < duration < duration<state_k+1>
            True;  duration == duration<state_k+1>

        """
        ret = []
        dur = int(self.duration)

        def fn(tid):

            if self.ts_type in ('rl', 'random') and \
                (dur == 1 or self.time_counter == 1) and \
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
                # New cycle.
                return True

            if self.ts_type == 'static':
                return dur in self.tls_durations[tid]
            elif self.ts_type == 'webster':
                return dur in self.webster_timings[tid]
            elif self.ts_type in ('rl', 'random'):
                if self.mdp_params.action_space == 'discrete':
                    # Discrete action space: TLS programs.
                    progid = self._current_rl_action()[tid]
                    return dur in self.programs[tid][progid]
                else:
                    # Continuous action space: phases durations.
                    return dur in self.signal_plans_continous[tid]
            else:
                raise ValueError(f'Unknown ts_type:{self.ts_type}')

        ret = [fn(tid) for tid in self.tls_ids]

        return tuple(ret)

    def apply_rl_actions(self, rl_actions):
        """ Specify the actions to be performed.

        Parameters:
        ----------
        rl_actions: dict or None
            actions to be performed

        """
        self.update_observation_space()

        if is_controller_periodic(self.ts_type):

            if self.ts_type in ('rl', 'random') and \
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

                    self.tsc.update(prev_state, prev_action, reward, state)

            if self.ts_type == 'webster':
                kernel_data = {nid: {p: build_vehicles(nid, data['incoming'], self.k.vehicle)
                                for p, data in self.tls_phases[nid].items()}
                                    for nid in self.tls_ids}
                self.webster_timings = self.tsc.act(kernel_data)

            if self.ts_type  == 'static':
                pass # Nothing to do here.

            self._apply_tsc_actions(self._periodic_control_actions())

        else:
            # Aperiodic controller.
            # TODO: Choose Phase is aperiodic.
            if self.ts_type == 'rl' and self.step_counter >= self.min_green - 1:
                # Get the number of the current cycle.
                N = int((self.step_counter + 1)/ 5)
                if (self.step_counter + 1) % 5  == 0:

                    print(f'apply_rl_actions#{self.step_counter}:')

                    # Every five time steps is a possible candidate for action
                    state = self.get_state()

                    if self.time_counter > 5:
                        import ipdb; ipdb.set_trace()

                    # Select new action.
                    if rl_actions is None:
                        this_actions = self._rl_actions(state)
                    else:
                        this_actions = rl_actions

                    if N > 0:

                        self.actions_log[N] = this_actions
                        self.states_log[N] = state
                        # RL-agent update.
                        reward = self.compute_reward(None)
                        prev_state = self.states_log[N - 1]
                        prev_action = self.actions_log[N - 1]
                        # TODO: Prevent agent from taking actions
                        # that violate constraints
                        
                        # TODO: Choose Phase
                        # Soft constraint on minimum and maximum
                        # -1o should be a very large penalty
                        def softr(x, y):
                            if x[1] < self.min_green + 5 and x[1] >= self.max_green: return -10
                            return y

                        soft_reward = {
                            tlid: softr(state[tlid], rwd) for tlid, rwd in reward.items()
                        }
                        self.tsc.update(prev_state, prev_action, soft_reward, state)
                else:
                    this_actions = self.actions_log[N - 1]

                # TODO: Choose Phases controller action mapping.
                # controller actions usually have twice as many phases.
                def fn(x):
                    if x[1] <= self.min_green: return True
                    return False

                def ctrl(x):
                    if x[1] < self.min_green: return 2 * x[0] + 1
                    return 2 * x[0]

                controller_actions = {
                    tlid: ctrl(sta)
                    for tlid, sta in self._cached_state.items() if fn(sta)
                }

                # Update traffic lights' control signals.
                self._apply_tsc_actions(controller_actions)

                self.update_active_phases(this_actions)

            if self.ts_type == 'max_pressure':
                controller_actions = self.tsc.act(self.observation_space, self.time_counter)

                # Update traffic lights' control signals.
                self._apply_tsc_actions(controller_actions)

            if self.ts_type == 'actuated':
                pass # Nothing to do here.

    def _apply_tsc_actions(self, controller_actions):
        """ For each TSC shift phase or keep phase.

        Parameters:
        ----------
        controller_actions: list<bool>
            False; keep state
            True; switch to next state

        """
        # TODO: Choose Phase
        # for i, tlid in enumerate(self.tls_ids):
        #     if controller_actions[i]:
        #         states = self.tls_states[tlid]
        #         next_state = states[controller_actions[i]]
        #         self.k.traffic_light.set_state(
        #             node_id=tlid, state=next_state)

        for i, tlid in enumerate(self.tls_ids):
            if controller_actions[tlid]:
                states = self.tls_states[tlid]
                self._tls_phase_indicator[tlid] = \
                        (self._tls_phase_indicator[tlid] + 1) % len(states)
                next_state = states[self._tls_phase_indicator[tlid]]
                self.k.traffic_light.set_state(
                    node_id=tlid, state=next_state)

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
        return self.reward(self.get_state())

    def reset(self):
        super(TrafficLightEnv, self).reset()
        self._reset()

    def _reset(self):
        # The 'duration' variable measures the elapsed time since
        # the beggining of the cycle, i.e. it measures (in seconds)
        # for how long the current configuration has been going on.
        self._duration_counter = -1
        self._duration = defaultdict(lambda : 0)
        self._active_phases = defaultdict(lambda : 0)

        # Choose Phase:
        # Always same inital state 0 (no yellow).
        self.min_green = 5
        self.max_green = 90

        
        # Stores the state index.
        self._tls_phase_indicator = {}
        for node_id in self.tls_ids:
            if self.ts_type != 'actuated':
                self._tls_phase_indicator[node_id] = 0
                s0 = self.tls_states[node_id][0]
                self.k.traffic_light.set_state(node_id=node_id, state=s0)

                self.actions_log[0] = {tlid: 0 for tlid in self.tls_ids}
                self.states_log[0] = self._cached_state

        # FIXME: Deprecate
        # Observation space.
        # self.observation_space.reset()
