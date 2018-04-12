from activation import *
import numpy as np
import types
from sgd import SGD

sgd = SGD()

class Layer(object):

    def __init__(self, dim=1, activation=sig):
        self.dim = dim
        self.activation = activation

    ''' Init w, b, and etc.
        w is a dim * input_dim matrix
        b is a dim array (1 * dim)
    '''
    def init_params(self, optimizer=sgd, input_dim=1):
    	self.optimizer = optimizer
        self.w = np.random.uniform(-np.sqrt(1. / self.dim), np.sqrt(1. / self.dim), (self.dim, input_dim))
        self.b = np.random.uniform(-np.sqrt(1. / self.dim), np.sqrt(1. / self.dim), self.dim)

    ''' Forward propagation:
            z = sum(w * x) + b
            a = activation(z)
        batchX is a batch_size * input_dim matrix
        batchZ is a batch_size * dim matrix
        batchA is a batch_size * dim matrix
    '''
    def forward(self, batchX):
        self.batchX = batchX
        self.batchZ = np.dot(batchX, self.w.T) + [self.b] * len(batchX)
        self.batchA = self.activation.primitive(self.batchZ)

        return self.batchA

    ''' Backward propagation
        next_* are properties of the next layer
        next_batchDz is a batch_size * next_dim matrix
        next_w is a next_dim * dim matrix, if it is None, 
               current layer is output layer
        derivative is a batch_size * dim matrix
        batchDz is a batch_size * dim matrix
    '''
    def backward(self, next_batchDz, next_w=None):
        if isinstance(next_w, types.NoneType):
            batchDa = next_batchDz
        else:
            batchDa = np.dot(next_batchDz, next_w)

        batchDaz = self.activation.derivative(self.batchZ, self.batchA)
        # Do hadamard product, which requres tmp and deri both are np.ndarray
        self.batchDz = batchDa * batchDaz

        return self.batchDz

    ''' Update w, b ant etc
    '''
    def update(self, lr=0.005):
    	# print "update weights in Layer"
    	# print "w:", self.w
    	# print "b:", self.b
        m = len(self.batchDz)
        for i in range(len(self.batchDz)):
            dz = np.array([self.batchDz[i]])
            x = np.array([self.batchX[i]])
            self.w -= lr / m * self.optimizer.update(np.dot(dz.T, x))
            self.b -= lr / m * self.optimizer.update(self.batchDz[i])
        # print "VVV"
        # print "w:", self.w
        # print "b:", self.b