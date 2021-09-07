import numpy as np

def batch_split(data, batch_size):
    n_batch = len(data) // batch_size
    return np.array_split(data, n_batch)

class InfinityDataGenerator:
    def __init__(self, x, y, batch_size=1):
        assert len(x) == len(y)
        self.i = 0

        self.x = batch_split(x, batch_size)
        self.y = batch_split(y, batch_size)   

    def reset(self):
        self.i = 0

    def __len__(self):
        return len(self.x)

    def __next__(self):
        retx = self.x[self.i]
        rety = self.y[self.i]

        self.i += 1
        if self.i >= len(self.x):
            self.i = 0

        return retx, rety

class InfinityMapDataGenerator:
    def __init__(self, data, map_fn=lambda x:x, batch_size=1):
        self.i = 0
        self.map_fn = map_fn

        self.data = batch_split(data, batch_size)  

    def reset(self):
        self.i = 0

    def __len__(self):
        return len(self.data)

    def __next__(self):
        data = self.data[self.i]

        retx = []
        rety = []
        for i in data:
            x, y = self.map_fn(i)
            
            retx.append(x)
            rety.append(y)

        self.i += 1
        if self.i >= len(self.data):
            self.i = 0

        return np.float32(retx), np.float32(rety)