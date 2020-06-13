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


class AgentWorker(multiprocessing.Process):
    
    def __init__(self, pipe):
        multiprocessing.Process.__init__(self)

        self.pipe = pipe
        #print(f'AgentWorker {self.name} - Pipe:', self.pipe)

    def run(self):

        #print(multiprocessing.get_context('spawn'))

        while True:
            msg = self.pipe.recv()
            if msg is None:
                # Poison pill means shutdown
                #print(f'{proc_name}: Exiting')
                break
            print(f'AgentWorker {self.name} - Received: {msg}')

            # Call method.
            ret = getattr(self, msg[0])(*msg[1])

            time.sleep(2)
            self.pipe.send(ret)

        return


class Agent(AgentWorker,AgentInterface):

    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)

    def init(self):
        print(f'Agent {self.name}: self.init()')
        self.A = tf.random.uniform((2,2))
        print(f'Agent {self.name}: self.A = {self.A}')
        return self.A

    def act(self):
        print(f'Agent {self.name}: self.act()')
        return f'acting {self.name}'

    def update(self, x, y, z):
        print(f'Agent {self.name}: self.update({x}, {y}, {z})')

    def save_checkpoint(self, path):
        print(f'Agent {self.name}: self.save_checkpoint({path})')

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        print(f'Agent {self.name}: self.load_checkpoint({chkpts_dir_path}, {chkpt_num})')


if __name__ == '__main__':

    NUM_CONSUMERS = 2

    multiprocessing.set_start_method('spawn')

    A = np.array([[2,3,4], [1,2,1]])

    # Create communication queues.
    pipes, consumers = {}, {}
    for i in range(NUM_CONSUMERS):
        comm_pipe = multiprocessing.Pipe()
        #print(f'MASTER - Comm pipe {comm_pipe}')

        consumers[i] = Agent(comm_pipe[1])
        pipes[i] = comm_pipe[0]

    #print('MASTER - Pipes:')
    #print(pipes)

    for i in consumers.keys():
        #print(f'MASTER - Started consumer {i}')
        consumers[i].start()

    #print('-'*50)

    """
        Init
    """
    for i in consumers.keys():
        pipes[i].send(('init', ()))

    for i in consumers.keys():
        msg = pipes[i].recv()
        print(f'MASTER - Received from {i}: {msg}')

    """
        Update
    """
    updates = [(345,34,-6), (4,4,-4)]
    for i in consumers.keys():
        pipes[i].send(('update', updates[i]))

    for i in consumers.keys():
        msg = pipes[i].recv()
        print(f'MASTER - Received from {i}: {msg}')

    """
        Act
    """
    for i in consumers.keys():
        pipes[i].send(('act', ()))

    for i in consumers.keys():
        msg = pipes[i].recv()
        print(f'MASTER - Received from {i}: {msg}')

    """
        End
    """
    # End consumers.
    for i in consumers.keys():
        pipes[i].send(None)
    
    print(f'MASTER ENDED')
