import multiprocessing
import time
import numpy as np

import tensorflow as tf

import abc


class AgentInterface(abc.ABC):

    @abc.abstractmethod
    def init(self, params, exp_path, name):
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
                break

            # Call method.
            ret = getattr(self, call[0])(*call[1])
            print(f'{self.name} - Received: {call}')

            # Fake work.
            time.sleep(5)

            # Send response.
            self.pipe.send(ret)
            print(f'{self.name} - Sent: {ret}')

        return


class Agent(AgentWorker,AgentInterface):

    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)

    def init(self):
        self.A = tf.random.uniform((2,2))
        return self.A

    def act(self, s):
        return f'Acting {self.name}'

    def update(self, x, y, z):
        return

    def save_checkpoint(self, path):
        return

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        return


class AgentClient(object):

    def __init__(self):
        # Create communication pipe.
        mp = multiprocessing.get_context('spawn')
        comm_pipe = mp.Pipe()

        self.agent = Agent(comm_pipe[1])
        self.pipe = comm_pipe[0]

        # Start agent.
        self.agent.start()

        # Initialize agent.
        args = ()
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

    def close(self):
        self.pipe.send(None)


class AgentsWrapper(object):

    def __init__(self, tids):

        self.tids = tids

        agents = {}

        # Create processes.
        for tid in self.tids:
            agents[tid] = AgentClient()

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

    def close(self):

        # Send request.
        for (tid, agent) in self.agents.items():
            agent.close()


if __name__ == '__main__':

    tids = ['gneJ0', 'gneJ1', 'gneJ2']

    agents_wrapper = AgentsWrapper(tids)

    s = {'gneJ0': (9,), 'gneJ1': (9,9), 'gneJ2': (9,9,9)}
    ret = agents_wrapper.act(s)
    print(ret)

    agents_wrapper.close()

    print(f'MAIN ENDED')
