import numpy as np

class Linear_layer():
    """
        Implements a linear layer
        inp_dim = No. of dimensions in input \n
        out_dim: No. of dimensions required in output \n
        bias :To use bias or not Default: True \n
        initialize: Type of initialization for the layer weights Default:'normal' \n
                   'normal'- from normal distribution \n
                   'xavier_unifor' - xavier uniform intialisation \n
                   'xavier_normal' - xavier normal initialisation \n
    """
    def __init__(self, inp_dim, out_dim, bias = True, initialize = 'normal'):
        self.name = "linear"
        self.parameters = True
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.is_bias = bias
        self.initialize = initialize
        self.initiate()
        self.set_buff_grad()
        self.message = "Added Linear layer to the model with input dim {} and output dim {}".format(self.inp_dim, self.out_dim)
        
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
    
    def __call__(self, inp, i):
        print('*'*15,'Going through layer {}->Linear layer'.format(i),'*'*15)
        try:
            out = np.matmul(inp, self.weight)
        except ValueError as err:
            raise ValueError(err)
        mess = "Input batch size->{}, Remaining Dimension->{} \nMultiplied with weight \
matrix of dimension {} \nOutput is of size {}".format(inp.shape[0], inp.shape[1],self.weight.shape, out.shape)
        print(mess)
        if self.is_bias:
            out += self.bias
            
        self.input = inp
        print('*' * 70)
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