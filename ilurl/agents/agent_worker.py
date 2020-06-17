import multiprocessing

mp = multiprocessing.get_context('spawn')

class AgentWorker(mp.Process):
    """
        This class abstracts remote method calling.
        It implements mp.Process.run() method: It
        receives incoming requests, processes them
        (using methods defined in the subclass) and
        sends the results back to the client.
    """
    
    def __init__(self, pipe):
        """ Instantiate.

            Parameters:
            ----------
            * pipe: communication pipe used to send/receive
                    messages to/from the client.

        """
        multiprocessing.Process.__init__(self)
        self.pipe = pipe

    def run(self):
        while True:
            call = self.pipe.recv()

            if call is None:
                # Terminate.
                break

            # Call method.
            ret = getattr(self, call[0])(*call[1])

            # Send response.
            self.pipe.send(ret)

        return
