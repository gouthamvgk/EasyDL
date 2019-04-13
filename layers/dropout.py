import numpy as np

class dropout():
    """
        Implements the dropout mechanism where neurons are dropped randomly \n
        keep_prob: Probability of retaining a neuron in the layer Default:0.5
    """
    def __init__(self, keep_prob=0.5):
        self.name="dropout"
        self.parameters = False
        self.prob = keep_prob
        self.dropout_mask = None
        self.message = "Added dropout layer to the model with keep_prob".format(self.prob)
        
    def __call__(self, inp,i, mode='train'):
        print('*'*15,'Going through layer {}->Dropout layer'.format(i),'*'*15)
        print('Input batch size->{}, Remaining dimension {}'.format(inp.shape[0], inp.shape[1:]))
        shape = inp.shape
        if (len(shape) > 2):
            raise ValueError('Dropout supports only linear layer with two dimensions(batch, features)')
        if mode == 'train':
            mask = np.random.binomial(1, self.prob, shape)
            out = mask * inp
            out /= self.prob
            self.dropout_mask = mask
            for i in range(shape[0]):
                print('In batch {}, {} out of {} neurons are retained'.format(i+1, np.sum(mask[i]), shape[1]))
            print('*' * 70)
            return out
        elif mode == 'test':
            print('This is test time so no dropout masking is carried out')
            print('*'*70)
            return inp
    
    def backward(self, inp_error):
        out = inp_error * self.dropout_mask
        self.dropout_mask = None
        return out