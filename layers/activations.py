import numpy as np

class sigmoid():
    def __init__(self):
        self.buffer = None
        self.parameters = False
        
    def __call__(self, inp):
        out = 1 / (1 + np.exp(-inp))
        self.buffer = out
        return out
    
    def backward(self, error):
        gate = self.buffer * (1-self.buffer)
        out = error * gate
        self.buffer = None
        return out
    
class tanh():
    def __init__(self):
        self.buffer = None
        self.parameters = False
        
    def __call__(self, inp):
        e_2x = np.exp(2 * inp)
        out = (e_2x-1)/(e_2x+1)
        self.buffer = out
        return out
    
    def backward(self, error):
        gate = 1 - (self.buffer ** 2)
        out = error * gate
        self.buffer = None
        return out
    
class relu():
    def __init__(self):
        self.buffer = None
        self.parameters = False
        
    def __call__(self, inp):
        self.buffer = inp > 0
        out = self.buffer * inp
        return out
    
    def backward(self, error):
        out = error * self.buffer
        self.buffer = None
        return out