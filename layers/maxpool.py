import numpy as np

class maxpool():
    """
        Does maxpooling over the given input \n
        kernel_size: kernel dimension for doing maxpool(supports only nxn)\n
        stride: stride for doing maxpooling Default:1 \n
        padding: Amount of padding to be added Default:0 \n
    """
    def __init__(self, kernel_size, stride=1, padding=0):
        self.name="maxpool"
        self.parameters = False
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.max_map = None
        self.input_shape = None
        self.message = "Added maxpool layer to model with kernel size {}, stride {} \
, pad {}".format(self.kernel_size, self.stride, self.pad)
        
    def check_find_dim(self, shape):
        if (len(shape) != 4):
            raise ValueError('Input has {} dimension instead of 4'.format(len(shape)))
        if (shape[2] != shape[3]):
            raise ValueError('Height and width of the input must be same')
        dim = (shape[-1] - self.kernel_size + (2*self.pad))/self.stride + 1
        if (dim).is_integer():
            return int(dim)
        else:
            raise ValueError('Input shape doesnt match with kernel size and stride')
            
    def find_map(self, inp, out_dim):
        out = np.zeros((inp.shape[0], out_dim, out_dim))
        max_map = np.zeros((inp.shape[0], out_dim, out_dim))
        (h,w) = (-1,-1)
        shape = inp.shape
        for i in range(0,shape[2]+1-self.kernel_size, self.stride):
            h += 1
            w = -1
            for j in range(0, shape[2]+1-self.kernel_size, self.stride):
                w += 1
                result = np.max(inp[:,i:i+self.kernel_size,j:j+self.kernel_size].reshape(shape[0], self.kernel_size*self.kernel_size), axis=1)
                out[:,h,w] = result
                indices = np.argmax(inp[:,i:i+self.kernel_size,j:j+self.kernel_size].reshape(shape[0], self.kernel_size*self.kernel_size), axis=1)
                max_map[:,h,w] = indices
        return out, max_map
        
    def __call__(self, inp,i):
        print('*'*15,'Going through layer {}->Maxpool layer'.format(i),'*'*15)
        out_dim = self.check_find_dim(inp.shape)
        self.input_shape = inp.shape
        out = np.zeros((inp.shape[0], inp.shape[1], out_dim, out_dim))
        max_map = np.zeros((inp.shape[0], inp.shape[1], out_dim, out_dim))
        inp = np.pad(inp, ((0,0),(0,0),(self.pad, self.pad),(self.pad,self.pad)),  'constant', constant_values=((0,0),))
        for k in range(inp.shape[1]):
            out[:,k,:,:], max_map[:,k,:,:] = self.find_map(inp[:,k,:,:], out_dim)
        self.max_map = max_map
        print('*' * 70)
        return out
    
    def backward(self, error):
        if (self.input_shape == None):
            raise ValueError('Backpropagation called without forward call')
        inp_grad = np.zeros(self.input_shape)
        inp_grad = np.pad(inp_grad, ((0,0),(0,0),(self.pad, self.pad),(self.pad,self.pad)),  'constant', constant_values=((0,0),))
        shape = inp_grad.shape
        for k in range(shape[1]):
            (h,w) = (-1,-1)
            for i in range(0,shape[2]+1-self.kernel_size, self.stride):
                h += 1
                w = -1
                for j in range(0, shape[2]+1-self.kernel_size, self.stride):
                    w += 1
                    temp = np.zeros((shape[0], self.kernel_size*self.kernel_size))
                    temp[list(range(0,shape[0])), list(map(lambda x:int(x), list(self.max_map[:,k,h,w])))] = error[:,k,h,w]
                    temp = temp.reshape(temp.shape[0], self.kernel_size, self.kernel_size)
                    inp_grad[:,k,i:i+self.kernel_size,j:j+self.kernel_size] += temp
        if self.pad > 0:
            inp_grad = inp_grad[:,:,0+self.pad:0-self.pad,0+self.pad:0-self.pad]
        self.max_map = None
        self.input_shape = None
        return inp_grad
                    
        
        
        
        
        
        
        
        
                
                
                