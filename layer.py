from activation import *
import numpy as np

class Layer:

	def __init__(self, dim=1, activation=sig):
		self.dim = dim
		self.activation = activation

	''' Init w, b, and etc.
	    w is a dim * input_dim matrix
	    b is a dim array (1 * dim)
	'''
	def init_params(self, input_dim=1):
		self.w = np.random.uniform(-np.sqrt(1. / d1), np.sqrt(1. / d1), (dim, input_dim))
		self.b = np.random.uniform(-np.sqrt(1. / d1), np.sqrt(1. / d1), dim)

	''' Forward propagation:
	        z = sum(w * x) + b
	        a = activation(z)
	    batchX is a batch_size * input_dim matrix
	    batchZ is a batch_size * dim matrix
	    batchA is a batch_size * dim matrix
	'''
	def forward(self, batchX):
		self.batchX = batchX
		self.batchZ = np.dot(batchX, self,w) + [self.b] * len(batchX)
		self.batchA = self.activation(self.batchZ)

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
		if w == None:
			batchDa = next_batchDz
		else:
			batchDa = np.dot(next_batchDz, next_w)
		batchDaz = self.activation.derivative(self.batchZ)
		# Do hadamard product, which requres tmp and deri both are np.ndarray
		self.batchDz = batchDa * batchDaz

		return self.batchDz

	''' Update w, b ant etc
	    batchDw is a dim * input_dim matrix
	'''
	def update(self, lr=0.005):
		m = len(self.batchDz)
		for i in range(len(self.batchDz)):
			self.w -= lr / m * np.dot(self.batchDz[i], self.batchX[i].T)
			self.b -= lr / m * self.batchDz[i]