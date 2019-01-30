import numpy as np
import warnings

class batch_norm1d():
    def __init__(self, num_features, momentum=0.1):
        self.parameters = True
        self.num_features = num_features
        self.momentum = momentum
        self.set_buff()
        self.set_grad()
        
    def set_buff(self):
        self.gamma = np.random.uniform(0,1,(1,self.num_features))
        self.beta = np.zeros((1, self.num_features))
        self.batch_mean = np.zeros((1, self.num_features))
        self.batch_var = np.ones((1,self.num_features))
        self.batch_size = 0
        self.batches_tracked = 0
    
    def set_grad(self):
        self.gamma_grad = np.random.uniform(0,1,(1,self.num_features))
        self.beta_grad = np.zeros((1, self.num_features))
        self.temp_inp = None
        self.temp_norm_inp = None
        self.temp_mean = None
        self.temp_var = None
        
    def __call__(self, inp, mode = 'train'):
        if mode == 'train':
            shape = inp.shape
            if shape[0] == 1:
                raise ValueError('Batch norm can be carried out with minimum of two datapoints')
            if self.batch_size != shape[0] and self.batches_tracked >= 1:
                warnings.warn('Current Batch size is not equal to previous train batch size which is not desirable')
            mean = np.sum(inp, axis = 0, keepdims=True)
            mean /= shape[0]
            var = np.sum((inp-mean)**2, axis=0, keepdims=True)
            var /= shape[0]
            inp_norm = (inp-mean)/np.sqrt(var+0.00005)
            out = (inp_norm * self.gamma) + self.beta
            self.temp_inp = inp
            self.temp_norm_inp = inp_norm
            self.temp_mean = mean
            self.temp_var = var
            
            self.batch_size = shape[0]
            self.batch_mean = self.momentum*self.batch_mean + (1-self.momentum) * mean
            self.batch_var = self.momentum*self.batch_var + (1-self.momentum) * var
            self.batches_tracked += 1
            return out
            
    def backward(self, norm_error):
        if norm_error.shape != self.temp_inp.shape:
            raise ValueError('Error in dimension between input and backpropagated error')
        norm_inp_grad = norm_error * self.gamma
        self.beta_grad += np.sum(norm_error, axis = 0, keepdims=True)
        self.gamma_grad += np.sum(norm_error*self.temp_norm_inp, axis = 0, keepdims=True)
        var_grad = norm_inp_grad * (self.temp_inp-self.temp_mean) * (-0.5) * ((self.temp_var + 0.00005)**(-1.5))
        var_grad = np.sum(var_grad, axis=0, keepdims=True)
        mean_grad = var_grad * (np.sum(-2*(self.temp_inp-self.temp_mean), axis=0, keepdims=True))/norm_error.shape[0]
        mean_grad += np.sum(norm_inp_grad * (-1/np.sqrt(self.temp_var+0.00005)), axis=0, keepdims=True)
        input_grad = norm_inp_grad * (1/np.sqrt(self.temp_var+0.00005))
        input_grad += ((2*(self.temp_inp-self.temp_mean))/norm_error.shape[0]) * var_grad
        input_grad += mean_grad/norm_error.shape[0]
        return input_grad
    
    def gradient_step(self,step_size):
        self.gamma -= step_size * self.gamma_grad
        self.beta -= step_size * self.beta_grad
        self.set_grad()
        