import numpy as np

class flatten():
    def __init__(self):
        self.size = None
        self.parameters = False
    def __call__(self, inp):
        size = inp.shape
        out = inp.reshape(size[0], np.product(size[1:]))
        self.size = size
        return out
    
    def backward(self, error):
        out = error.reshape(self.size)
        self.size = None
        return out