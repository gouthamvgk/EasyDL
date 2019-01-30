import numpy as np

class Linear_layer():
    def __init__(self, inp_dim, out_dim, bias = True, initialize = 'normal'):
        self.parameters = True
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.is_bias = bias
        self.initialize = initialize
        self.initiate()
        self.set_buff_grad()
        
    def initiate(self):
        if (self.initialize == 'normal'):
            self.weight = np.random.standard_normal(size = (self.inp_dim, self.out_dim))
        elif(self.initialize == 'xavier_uniform'):
            bound = np.sqrt(6/(self.inp_dim+self.out_dim))
            self.weight = np.random.uniform(low = -bound, high = bound, size=(self.inp_dim, self.out_dim))
        elif(self.initialize == 'xavier_normal'):
            std = np.sqrt(2/(self.inp_dim+self.out_dim))
            self.weight = np.random.normal(loc = 0, scale=std, size=(self.inp_dim, self.out_dim))
            
        if self.is_bias:
            self.bias = np.zeros((1, self.out_dim))
            
    def set_buff_grad(self):
        self.input = None
        self.weight_grad = np.zeros((self.inp_dim, self.out_dim))

        if self.is_bias:
            self.bias_grad = np.zeros((1,self.out_dim))
    
    def __call__(self, inp):
        out = np.matmul(inp, self.weight)
        if self.is_bias:
            out += self.bias
            
        self.input = inp
        return out
    
    def backward(self, error):
        temp = np.matmul(self.input.T, error)
        temp /= error.shape[0]
        self.weight_grad += temp
        if self.is_bias:
            temp = np.sum(error, axis = 0, keepdims = True)
            temp /= error.shape[0]
            self.bias_grad += temp
            
        inp_error = np.matmul(error, self.weight.T)
        return inp_error
    
    def gradient_step(self, step_size):
        self.weight -= step_size * self.weight_grad
        if self.is_bias:
            self.bias -= step_size * self.bias_grad
            
        self.set_buff_grad()