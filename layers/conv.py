
import numpy as np
import time
class Convolution():
    """
    Peforms convolution operation over the input.\n
    inp_fea: No. of channels in input \n
    out_fea: No. of channels required in output \n
    kernel: kernel dimension(supports only nxn) \n
    stride: stride for convolution\n
    pad_type: type of padding\n
        'same' - pads necessarily for same shape in output(dimensions should match)\n
        'valid'- pads according to given value\n
    pad: Amound of padding. Necessary only if pad_type='valid' \n
    bias: To use bias or not for filter. Default=True\n
    """
    def __init__(self, inp_fea, out_fea, kernel, stride, pad_type, pad = None, bias = True):
        self.name = "conv"
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
        self.message = "Added conv layer to the model with {}\
  inp channels, {} out channels, kernel size {}, stride {} and {}\
 pad".format(self.inp_fea, self.out_fea, self.kernel_size, self.stride, self.pad_type)
        
        
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
            print("Padding type is 'same' (output size same as input)...")
            print('Calculating pad value using formula--->(feature_size * (stride-1) + kernel_size - stride)/2')
            print('({} * ({}-1) + {} - {})/2 ------>same pad value= {}'.format(fea_size, self.stride, self.kernel_size, self.stride, self.pad))
            print('-'*60)
        else:
            raise ValueError('Same convolution cannot be performed for given feature dimension')
            
    def find_out_dim(self, fea_size, same_pad):
        out  = ((fea_size + 2*self.pad - self.kernel_size)/self.stride) + 1
        if (out).is_integer():
            print("Output feature map size is determined by kernel, stride and pad")
            print("Calculating output feature map size using formula--->((feature_size + 2*pad - kernel_size)/stride) + 1")
            print("(({} + 2*{} - {})/{}) + 1------->out feature size = {}".format(fea_size, self.pad, self.kernel_size, self.stride, out))
            if same_pad:
                print("It is same padding layer so input and output is of same size")
            print('-'*60)
            return int(out)
        else:
            raise ValueError('Given feature size doesnt suit the layer parameters')
     
    def convolution(self, shape, fil, inp, out_dim, fil_no, forward = True, print_=True):
        fil = fil.reshape(1,fil.shape[0]*fil.shape[1]*fil.shape[2])
        temp_out = np.zeros((inp.shape[0], 1, out_dim, out_dim))
        temp_fil = np.repeat(fil, repeats = inp.shape[0], axis = 0)
        (h,w) = (-1,-1)
        inp_shape = inp.shape
        for i in range(0,shape[2]+1-self.kernel_size, self.stride):
            h += 1
            w = -1
            limit = 0
            for j in range(0, shape[2]+1-self.kernel_size, self.stride):
                w += 1
                temp_inp = inp[:,:,i:i+self.kernel_size,j:j+self.kernel_size].reshape(inp_shape[0], inp_shape[1]*self.kernel_size*self.kernel_size)
                if print_ and limit < 3:
                    print('Input[:,:,{}:{},{}:{}] is convolved with the filter {} to give \
output[:,{},{},{}]'.format(i,i+self.kernel_size, j,j+self.kernel_size, fil_no+1, fil_no, h,w))
                elif print_ and limit >=3 and limit < 6:
                    print(' '*40, '.', ' '*40)
                out = temp_inp * temp_fil
                if self.is_bias and forward:
                    out += self.bias[fil_no]
                out = np.sum(out, axis = 1, keepdims=True)
                temp_out[:,:,h,w] = out
                limit += 1
            if print_:
                print('-'*50)
        temp_out = np.squeeze(temp_out, axis = 1)
        return temp_out      
                
        
    def __call__(self, inp, i, verb_neat=True):
        print('*'*15,'Going through layer {}->conv layer'.format(i),'*'*15)
        shape = inp.shape
        if (shape[1] != self.inp_fea):
            raise ValueError('No. of input channels is not suited for the layer')
        if (shape[2] != shape[3]):
            raise ValueError('Height and width of the input must be same')
        if self.pad_type == 'same':
            try:
                self.find_same_pad(shape[2])
            except ValueError as err:
                raise ValueError(err)
        try:
            out_dim = self.find_out_dim(shape[2], (self.pad_type=='same'))
        except ValueError as err:
            raise ValueError(err)
        print('Input Feature map shape before padding---->{}'.format(shape[1:]))
        inp = np.pad(inp, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant', constant_values=((0,0),))
        shape = inp.shape
        print('Input Feature map shape after padding for conv operation----->{}'.format(shape[1:]))
        output = np.zeros((shape[0], self.out_fea, out_dim, out_dim))
        print('Dimension of output feature map----->{}'.format(output.shape[1:]))
        print('-'*60)
        print('There are {} channels in the output'.format(self.out_fea))
        print('Convolution at each position is done as follows...')
        print('K th filter of dimension {} is flattened into {}...'.format(self.filters[0].shape, (1,np.product(self.filters[0].shape))))
        print('A window of input of dimension {} is flattened into {}...'.format(self.filters[0].shape, (1,np.product(self.filters[0].shape))))
        print('Both these are dot producted to form output[:,k,i,j]')
        print('*' * 70)
        for k in range(self.out_fea):
            print('Calculating channel {} of the output...'.format(k+1))
            print('Filter {} of dims {} is being convolved with padded input of dims {}'.format(k+1, self.filters[k].shape, shape[1:]))
            if not verb_neat:
                output[:,k,:,:] = self.convolution(shape, self.filters[k], inp, out_dim, fil_no=k, print_=True)
                print('x'*50)
            elif verb_neat and (k <=1 or k==(self.out_fea-1)):
                output[:,k,:,:] = self.convolution(shape, self.filters[k], inp, out_dim, fil_no=k, print_=True)
                print('x'*50)
            elif verb_neat:
                output[:,k,:,:] = self.convolution(shape, self.filters[k], inp, out_dim, fil_no=k, print_=False)
                print(' '*40, '.', ' '*40)
                print(' '*40, '.', ' '*40)
                print(' '*40, '.', ' '*40)
                print('x'*50)
                time.sleep(1)
        self.output_size = output.shape
        self.input = inp
        print('*' * 70)
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
            inp_grad[:,k,:,:] = self.convolution(shape, flipped_kernel[:,k,:,:],temp_error, self.input.shape[-1], fil_no=k,forward = False)
        return inp_grad 
            
    def gradient_step(self, step_size):
        self.filters -= self.filters_grad * step_size
        if self.is_bias:
            self.bias -= self.bias_grad * step_size
        self.set_buff_grad()