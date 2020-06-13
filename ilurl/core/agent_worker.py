import multiprocessing

mp = multiprocessing.get_context('spawn')

class AgentWorker(mp.Process):
    
    def __init__(self, pipe):
        multiprocessing.Process.__init__(self)

        self.pipe = pipe
        #print(f'AgentWorker {self.name} - Pipe:', self.pipe)

    def run(self):

        #print(multiprocessing.get_context('spawn'))

        while True:
            msg = self.pipe.recv()
            if msg is None:
                #print(f'{proc_name}: Exiting')
                break
            #print(f'{self.name} - Received: {msg}')

            # Call method.
            ret = getattr(self, msg[0])(*msg[1])
            self.pipe.send(ret)

        return
