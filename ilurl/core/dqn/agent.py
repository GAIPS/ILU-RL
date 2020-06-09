import os
import numpy as np

from ilurl.core.meta import MetaAgent
from ilurl.utils.default_logger import make_default_logger

import dm_env
import acme
from acme import specs
from ilurl.core.dqn import acme_agent
from acme import networks


class DQN(object, metaclass=MetaAgent):
    """
        DQN agent.
    """

    def __init__(self, params, exp_path, name):
        """Instantiate DQN agent.

        PARAMETERS
        ----------
        * params: ilurl.core.params.DQNParams object.
            object containing DQN parameters.

        * name: str

        """
        self.name = name

        # Whether learning stopped.
        self.stop = False

        observation_spec = specs.Array(shape=(params.states.rank,),
                                  dtype=np.float32,
                                  name='obs'
        )
        action_spec = specs.DiscreteArray(dtype=int,
                                          num_values=params.actions.depth,
                                          name="action"
        )
        reward_spec = specs.Array(shape=(),
                                  dtype=float,
                                  name='reward'
        ) # Default.
        discount_spec = specs.BoundedArray(shape=(),
                                           dtype=float,
                                           minimum=0.,
                                           maximum=1.,
                                           name='discount'
        ) # Default.
 
        env_spec = specs.EnvironmentSpec(observations=observation_spec,
                                          actions=action_spec,
                                          rewards=reward_spec,
                                          discounts=discount_spec)

        # Logger.
        dir_path = f'{exp_path}/logs/{self.name}'
        self._logger = make_default_logger(directory=dir_path, label=self.name)

        agent_logger = make_default_logger(directory=dir_path, label=f'{self.name}-learning')
        network = networks.duelling.DuellingMLP(num_actions=env_spec.actions.num_values,
                                                hidden_sizes=[8])
        self.agent = acme_agent.DQN(env_spec, network, logger=agent_logger)

        # Observations counter.
        self._obs_counter = 0

    @property
    def stop(self):
        """Stops learning."""
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop

    def act(self, s):
        """
        Specify the actions to be performed by the RL agent(s).

        Parameters:
        ----------
        * s: tuple
            state representation.

        """
        s = np.array(s, dtype=np.float32)

        # Make first observation.
        if self._obs_counter == 0:
            t_1 = dm_env.restart(s)
            self.agent.observe_first(t_1)

        # Select action.
        if self.stop:
            action = self.agent.deterministic_action(s)
        else:
            action = self.agent.select_action(s)

        self._obs_counter += 1

        return action

    def update(self, _, a, r, s1):
        """
        Performs a learning update step.

        Parameters:
        ----------
        * a: int
            action.

        * r: float
            reward.

        * s1: tuple
            state representation.

        """
        if not self.stop:

            s1 = np.array(s1, dtype=np.float32)
            timestep = dm_env.transition(reward=r, observation=s1)

            self.agent.observe(a, timestep)
            self.agent.update()

        # Log values.
        if self._logger:
            values = {
                'step': self._obs_counter,
                'action': a,
                'reward': r,
            }
            self._logger.write(values)

    def save_checkpoint(self, path):
        """
        Save model's weights.

        Parameters:
        ----------
        * path: str 
            path to save directory.

        """
        checkpoint_file = "{0}/checkpoints/{1}/{2}.chkpt".format(
            path, self._obs_counter, self.name)

        print('SAVED')
        print(checkpoint_file)

        self.agent.save(checkpoint_file)

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        """
        Loads model's weights from file.
 
        Parameters:
        ----------
        * chkpts_dir_path: str
            path to checkpoint's directory.

        * chkpt_num: int
            the number of the checkpoint to load.

        """
        chkpt_path = '{0}/{1}/{2}.chkpt'.format(chkpts_dir_path,
                                                    chkpt_num,
                                                    self.name)

        print('LOADED')
        print(chkpt_path)
        self.agent.load(chkpt_path)

    def setup_logger(self, path):
        """
        Setup train logger.
 
        Args:
        ----
        * path: str 
            path to log directory.

        """
        pass
        """ dir_path = f'{path}/logs/{self.name}'
        self.logger = make_default_logger(directory=dir_path, label=self.name) """