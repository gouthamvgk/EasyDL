import numpy as np
from layers.conv import Convolution
from layers.reshape import reshape
from layers.batch_norm1d import batch_norm1d
from layers.dropout import dropout
from layers.maxpool import maxpool
from layers.flatten import flatten
from layers.linear import Linear_layer

class model():
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
        print(layer.message+" as layer {}".format(len(self.layers)))
        print('*'*60)
        
    def __call__(self, inp, neat=True):
        print('*'*70)
        print('No. of batches in input {}\nInput is of dimension{}'.format(inp.shape[0], inp.shape[1:]))
        print('*'*70)
        out = inp
        for i,k in enumerate(self.layers):
            if k.name == 'conv':
                try:
                    out = k(out, i+1, neat)
                except ValueError as err:
                    raise ValueError(err)
            else:
                try:
                    out = k(out, i+1)
                except ValueError as err:
                    raise ValueError(err)
        return out