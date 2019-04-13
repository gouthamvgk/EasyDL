from layers.conv import Convolution
import numpy as np
from layers.maxpool import maxpool
from net import model
e = model()
e.add_layer(Convolution(3,10,5,2,'same'))
e.add_layer(maxpool(3,1))
e(np.random.randn(2,3,10,10))
