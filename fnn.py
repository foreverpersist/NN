from loss import *
from sga import *

losses = {'mse': mse, 'ce': ce}

optimizers = {'sgd': sgd, "momentum": momentum}

class FNN:
	''' Full-connection Neural Network
	    It is a simple network with multiple layers
	'''
	def __init__(self):
		self.layers = []

	''' Forward-progation
	    batchX is a batch_size * M matrix
	    self.batchA a batch_size * N matrix
	'''
	def forward(self, batchX):
		batchA = bacthX
		for layer in self.layers:
			batchA = layer.forward(batchA)

		self.batchA = self.output.forward(batchA)

		return self.batchA

	def bptt(self, x, y, lr):
		# Calculate @C/@aL
		delta_loss = self.backward()

		batchDz = self.output.backward(delta_loss)
		self.output.update(lr)
		
		last_layer = self.output
		layer_size = len(self.layers)
		for k in range(layer_size):
			i = layer_size - 1 - k
			layer = self.layers[i]
			batchDz = layer.backward(batchDz, last_layer.w)
			layer.update(lr)
			last_layer = layer

	# Backward-progation
	# Calculate  @C/@aL
	def backward(self, ):
		return self.delta(self.batchA, self.trainY)

	# Add a hidden layer
	def add(self, layer):
		self.layers.append(layer)

	# Set output layer
	def output(self, layer):
		self.output = layer

	def compile(self, loss='mse', optimizer='sgd'):
		self.loss = losses[loss]
		self.optimizer = optimizers[optimizer]

	def fit(self, trainX, trainY, \
		lr=0.005, batch_size=1, epochs=30, shuffle=False, verbose=2):
		self.trainX = self.trainY
		self.trainY = self.trainY
		pass

	def predict(self, X):
		pass