import numpy as np

class dropout():
    def __init__(self, keep_prob=0.5):
        self.parameters = False
        self.prob = keep_prob
        self.dropout_mask = None
        
    def __call__(self, inp, mode='train'):
        shape = inp.shape
        if (len(shape) > 2):
            raise ValueError('Dropout supports only linear layer with two dimensions(batch, features)')
        if mode == 'train':
            mask = np.random.binomial(1, self.prob, shape)
            out = mask * inp
            out /= self.prob
            self.dropout_mask = mask
            return out
        elif mode == 'test':
            return inp
    
    def backward(self, inp_error):
        out = inp_error * self.dropout_mask
        self.dropout_mask = None
        return out