import multiprocessing

mp = multiprocessing.get_context('spawn')

class AgentClient(object):
    """
        This class abstracts remote method calling.
        It provides an interface to connect with
        agent instances that are running on separate
        processes.
    """

    def __init__(self, agent_cls, params):
        """ Instantiate.

            Parameters:
            ----------
            * agent_cls: agent's class for instantiation.
            * params: agent's parameters.

            Attributes:
            ----------
            * pipe: communication pipe used to send/receive
                    messages to/from the agent class.
            * agent: agent's instance.

        """

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



    # TODO: make decorator (or simplify = *args) the stuff below.

    def get_network(self):
        args = ()
        func_call = ('get_network', args)
        self.pipe.send(func_call)


    def get_stop(self):
        args = ()
        func_call = ('get_stop', args)
        self.pipe.send(func_call)

    def set_stop(self, stop):
        args = (stop,)
        func_call = ('set_stop', args)
        self.pipe.send(func_call)

    def act(self, state):
        args = (state,)
        func_call = ('act', args)
        self.pipe.send(func_call)

    def update(self, s, a, r, s1):
        args = (s,a,r,s1)
        func_call = ('update', args)
        self.pipe.send(func_call)

    def save_checkpoint(self, path):
        args = (path,)
        func_call = ('save_checkpoint', args)
        self.pipe.send(func_call)

    def load_checkpoint(self, chkpts_dir_path, chkpt_num):
        args = (chkpts_dir_path, chkpt_num)
        func_call = ('load_checkpoint', args)
        self.pipe.send(func_call)

    def receive(self):
        """
            Receives message from agent.
        """
        return self.pipe.recv()

    def terminate(self):
        """
            Terminates and waits for agent's process execution.
        """
        self.pipe.send(None)
        self.agent.join()
