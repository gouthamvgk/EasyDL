import numpy as np
import warnings

class reshape():
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.parameters = False
    
    def __call__(self, inp, out_shape):
        inp_shape = inp.shape
        if (np.product(inp_shape) != np.product(out_shape)):
            raise ValueError('Reshape dimensions doesnt match with input dimension')
        if (inp_shape[0] != out_shape[0]):
            warnings.warn('Dimension 1 usually corresponds to batch size, reshaping is carried out with respect to that dimension')
        self.input_shape = inp_shape
        self.output_shape = tuple(out_shape)
        return inp.reshape(out_shape)
    
    def backward(self, error):
        if (self.output_shape != error.shape):
            raise ValueError('Error dimension doesnt match with required dimension')
        temp = error.reshape(self.input_shape)
        self.input_shape = None
        return temp