import numpy as np

class flatten():
    """
    Flattens the given input into one dimension. i.e input of shape (B,a,b,c,..)
    is converted to array of shape(N, axbxc...)
    """
    def __init__(self):
        self.name="flatten"
        self.size = None
        self.parameters = False
        self.message = "Added flatten layer to the model"
    def __call__(self, inp, i):
        print('*'*15,'Going through layer {}-> Flatten layer'.format(i),'*'*15)
        size = inp.shape
        print('Input batch size->{}, Remaining Dimension->{}'.format(size[0], size[1:]))
        out = inp.reshape(size[0], np.product(size[1:]))
        print('Flattened to output of shape {}'.format(out.shape))
        print('i.e Dimension {} converted to single dimension of size {}'.format(size[1:], out.shape[1]))
        self.size = size
        print('*' * 70)
        return out
    
    def backward(self, error):
        out = error.reshape(self.size)
        self.size = None
        return out