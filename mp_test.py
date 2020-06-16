import abc
import time
import functools
import multiprocessing
from typing import Callable, List

import tensorflow as tf


class AgentInterface(abc.ABC):
    """
        Single agent interface.
    """

    @abc.abstractmethod
    def init(self, params):
        """ Init """

    @abc.abstractmethod
    def act(self, s):
        """ Act """

    @abc.abstractmethod
    def update(self, s, a, r, s1): 
        """ Update """

    @abc.abstractmethod
    def save_checkpoint(self, path):
        """ Save checkpoint """

    @abc.abstractmethod
    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        """ Load checkpoint """


mp = multiprocessing.get_context('spawn')
class AgentWorker(mp.Process):
    
    def __init__(self, pipe):
        mp.Process.__init__(self)
        self.pipe = pipe

    def run(self):
        #print(multiprocessing.get_context('spawn'))

        while True:
            call = self.pipe.recv()

            if call is None:
                # Terminate.
                break

            # Call method.
            # print(f'{self.name} - Received: {call}')
            ret = getattr(self, call[0])(*call[1])

            # Fake work.
            time.sleep(3)

            # Send response.
            self.pipe.send(ret)
            # print(f'{self.name} - Sent: {ret}')

        return


class AgentsFactory(object):

    # Internal registry.
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: AgentInterface) -> Callable:
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def types(cls) -> List:
        return list(cls.registry.keys())

    @classmethod
    def get(cls, name : str):
        if name not in list(cls.registry.keys()):
            raise ValueError("unknown agent type.")
        return cls.registry[name]


class QLParams(object):
    def __init__(self, x):
        self.x = x


class DQNParams(object):
    def __init__(self, y):
        self.y = y


@AgentsFactory.register('QL')
class QL(AgentWorker,AgentInterface):

    def __init__(self, *args, **kwargs):
        super(QL, self).__init__(*args, **kwargs)

    def init(self, params):
        self.x = params.x
        self.A = tf.random.uniform((4,4))
        return self.A

    def act(self, s):
        return f'Acting {self.name}'

    def update(self, x, y, z):
        return f'Updating {self.name}'

    def save_checkpoint(self, path):
        return f'Saving {self.name}'

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        return f'Loading {self.name}'


@AgentsFactory.register('DQN')
class DQN(AgentWorker,AgentInterface):

    def __init__(self, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)

    def init(self, params):
        self.y = params.y
        self.A = tf.random.uniform((2,2))
        return self.A

    def act(self, s):
        return f'Acting {self.name}'

    def update(self, x, y, z):
        return f'Updating {self.name}'

    def save_checkpoint(self, path):
        return f'Saving {self.name}'

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        return f'Loading {self.name}'


##############################################################


class MultiAgentSystemInterface(abc.ABC):
    """
        Multi-agent system interface.
    """

    @abc.abstractmethod
    def act(self, s):
        """ Act """

    @abc.abstractmethod
    def update(self, s, a, r, s1): 
        """ Update """

    @abc.abstractmethod
    def save_checkpoint(self, path):
        """ Save checkpoint """

    @abc.abstractmethod
    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        """ Load checkpoint """


class AgentClient(object):

    def __init__(self, agent_cls, params):

        # Create communication pipe.
        mp = multiprocessing.get_context('spawn')
        comm_pipe = mp.Pipe()

        self.agent = agent_cls(comm_pipe[1])
        self.pipe = comm_pipe[0]

        # Start agent.
        self.agent.start()

        # Initialize agent.
        args = (params,)
        func_call = ('init', args)
        self.pipe.send(func_call)

        # Synchronize.
        self.pipe.recv()

    def act(self, state):
        args = (state,)
        func_call = ('act', args)
        self.pipe.send(func_call)

    def receive(self):
        return self.pipe.recv()

    def terminate(self):
        self.pipe.send(None)


class DecentralizedMAS(MultiAgentSystemInterface):

    def __init__(self, tids):

        self.tids = tids

        agents = {}

        # Create processes.
        for tid in self.tids:
            #agents[tid] = AgentsFactory.create('DQN', DQNParams(67))
            agents[tid] = AgentClient(AgentsFactory.get('DQN'), DQNParams(39))

        self.agents = agents
        print(self.agents)

    def act(self, state):

        # Send request.
        for (tid, agent) in self.agents.items():
            agent.act(state[tid])

        # Retrieve request.
        choices = {}
        for (tid, agent) in self.agents.items():
            choices[tid] = agent.receive()

        return choices
    
    def update(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def terminate(self):

        # Send terminate request for each agent.
        for (tid, agent) in self.agents.items():
            agent.terminate()

if __name__ == '__main__':

    tids = ['gneJ0', 'gneJ1', 'gneJ2']

    print(AgentsFactory.types())

    agents_wrapper = DecentralizedMAS(tids)

    s = {'gneJ0': (9,), 'gneJ1': (9,9), 'gneJ2': (9,9,9)}
    ret = agents_wrapper.act(s)
    print(ret)

    agents_wrapper.terminate()


    print(f'MAIN ENDED')