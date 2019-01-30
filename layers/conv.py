
import numpy as np

class Convolution():
    def __init__(self, inp_fea, out_fea, kernel, stride, pad_type, pad = None, bias = True):
        self.parameters = True
        self.inp_fea = inp_fea
        self.out_fea = out_fea
        self.kernel_size = kernel
        self.stride = stride
        self.pad_type = pad_type
        self.pad = pad
        self.is_bias = bias
        if self.stride > self.kernel_size:
            raise ValueError('Stride value greater than kernel size')
        if self.pad_type == 'valid' and self.pad == None:
            raise ValueError('Padding dim must be specified for valid convolution')
        self.initiate()
        self.set_buff_grad()
        
    def initiate(self):
        self.filters = np.random.standard_normal(size=(self.out_fea, self.inp_fea, self.kernel_size, self.kernel_size))
        if self.is_bias:
            self.bias = np.zeros((self.out_fea, 1))
            
    def set_buff_grad(self):
        self.filters_grad = np.zeros((self.out_fea, self.inp_fea, self.kernel_size, self.kernel_size))
        if self.is_bias:
            self.bias_grad = np.zeros((self.out_fea, 1))
        self.input = None
        self.output_size = None
        
    def find_same_pad(self, fea_size):
        pad = (fea_size * (self.stride-1) + self.kernel_size - self.stride)/2
        if (pad).is_integer():
            self.pad = int(pad)
        else:
            raise ValueError('Same convolution cannot be performed for given feature dimension')
            
    def find_out_dim(self, fea_size):
        out  = ((fea_size + 2*self.pad - self.kernel_size)/self.stride) + 1
        if (out).is_integer():
            return int(out)
        else:
            raise ValueError('Given feature size doesnt suit the layer parameters')
     
    def convolution(self, shape, fil, inp, out_dim, forward = True):
        fil = fil.reshape(1,fil.shape[0]*fil.shape[1]*fil.shape[2])
        temp_out = np.zeros((inp.shape[0], 1, out_dim, out_dim))
        temp_fil = np.repeat(fil, repeats = inp.shape[0], axis = 0)
        (h,w) = (-1,-1)
        inp_shape = inp.shape
        for i in range(0,shape[2]+1-self.kernel_size, self.stride):
            h += 1
            w = -1
            for j in range(0, shape[2]+1-self.kernel_size, self.stride):
                w += 1
                temp_inp = inp[:,:,i:i+self.kernel_size,j:j+self.kernel_size].reshape(inp_shape[0], inp_shape[1]*self.kernel_size*self.kernel_size)
                out = temp_inp * temp_fil
                if self.is_bias and forward:
                    out += self.bias
                out = np.sum(out, axis = 1, keepdims=True)
                temp_out[:,:,h,w] = out
        temp_out = np.squeeze(temp_out, axis = 1)
        return temp_out      
                
        
    def __call__(self, inp):
        shape = inp.shape
        if (shape[1] != self.inp_fea):
            raise ValueError('No. of input channels is not suited for the layer')
        if (shape[2] != shape[3]):
            raise ValueError('Height and width of the input must be same')
        if self.pad_type == 'same':
            try:
                self.find_same_pad(shape[2])
            except ValueError as err:
                print(err)
                return
        try:
            out_dim = self.find_out_dim(shape[2])
        except ValueError as err:
            print(err)
            return
        inp = np.pad(inp, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant', constant_values=((0,0),))
        shape = inp.shape
        output = np.zeros((shape[0], self.out_fea, out_dim, out_dim))
        for k in range(self.out_fea):
            output[:,k,:,:] = self.convolution(shape, self.filters[k], inp, out_dim)
        self.output_size = output.shape
        self.input = inp
        return output
        
    def backward_conv(self, inp, error):
        temp_fil = np.zeros((1, self.kernel_size, self.kernel_size))
        shape = inp.shape
        temp = error.reshape(error.shape[0], error.shape[1]*error.shape[2])
        (h,w) = (-1,-1)
        for i in range(0,shape[2]+1-error.shape[-1], self.stride):
            h += 1
            w = -1
            for j in range(0, shape[2]+1-error.shape[-1], self.stride):
                w += 1
                temp_inp = inp[:,i:i+error.shape[-1],j:j+error.shape[-1]]
                temp_inp = temp_inp.reshape(temp_inp.shape[0], temp_inp.shape[1]*temp_inp.shape[2])
                out = temp * temp_inp
                out = np.sum(out, axis=1, keepdims=True)
                out = float(np.sum(out, axis=0)) / shape[0]
                temp_fil[:,h,w] = out
        return temp_fil
    
    def backward(self,error):
        if (self.output_size == None):
            raise ValueError('Backward step called without forward propagating')
        if (self.output_size != error.shape):
            raise ValueError('Error dimension mismatch occured')
        for i in range(self.out_fea):
            temp_fil = np.zeros((1,self.inp_fea, self.kernel_size, self.kernel_size))
            for k in range(self.inp_fea):
                temp_fil[:,k,:,:] = self.backward_conv(self.input[:,k,:,:], error[:,i,:,:])
            self.filters_grad[i,:,:,:] += np.squeeze(temp_fil, axis=0)
        if self.is_bias:
            temp = np.sum(error.reshape(error.shape[0], error.shape[1], error.shape[2]*error.shape[3]), axis = 2)
            temp = np.sum(temp, axis = 0, keepdims=True)
            temp = temp.reshape(temp.shape[1], 1)
            temp /= temp.shape[0]
            self.bias_grad += temp
            
        inp_grad = np.zeros(self.input.shape)
        flipped_kernel = np.flip(self.filters, axis=(2,3))  
        temp_dim = error.shape[-1] + (error.shape[-1] - 1) * (self.stride-1)
        temp_error = np.zeros((error.shape[0], error.shape[1], temp_dim, temp_dim))
        (h,w) = (-1,-1)
        for i in range(error.shape[-1]):
            h += 1
            w = -1
            for k in range(error.shape[-1]):
                w += 1
                temp_error[:,:,h,w] = error[:,:,i,k]
                w += self.stride -1
            h += self.stride - 1 
        temp_error = np.pad(temp_error, ((0,0),(0,0),(self.kernel_size-1,self.kernel_size-1),(self.kernel_size-1,self.kernel_size-1)), 'constant', constant_values=((0,0),))
        shape = temp_error.shape
        for k in range(self.inp_fea):
            inp_grad[:,k,:,:] = self.convolution(shape, flipped_kernel[:,k,:,:],temp_error, self.input.shape[-1], forward = False)
        return inp_grad 
            
    def gradient_step(self, step_size):
        self.filters -= self.filters_grad * step_size
        if self.is_bias:
            self.bias -= self.bias_grad * step_size
        self.set_buff_grad()