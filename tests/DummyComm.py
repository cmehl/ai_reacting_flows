class DummyComm:
    def Get_rank(self):
        return 0
    def Get_size(self):
        return 1
    def Barrier(self):
        pass
    def bcast(self, data, root=0):
        return data
    def gather(self, data, root=0):
        return [data]
    def scatter(self, data, root=0):
        return data[0]
    def allreduce(self, data, op=None):
        return data
    def allgather(self, data, root=0):
        return [data]