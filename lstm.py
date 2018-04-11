from activation import *
from layer import *
import numpy as np


class LstmCell:
	''' There are four types gates in LSTM:
	        InputGate:     i =  sig (ui * x + wi * h + bi)
	        ForgetGate:    f =  sig (uf * x + wf * h + bf)
	        OutputGate:    o =  sig (uo * x + wo * h + bo)
	        CandidateGate: ci = tanh(uc * x + wc * h + bc)
	    Update state and output:
	    	State:  c = f * c + i * ci
	    	Output: h = o * tanh(c)
	'''
	def __init__(self):
		self.i_activation = sig
		self.f_activation = sig
		self.o_activation = sig
		self.ci_activation = tanh
		self.activation = tanh

	''' Note: d here is just a number for initializing params,
	          you can use any number except 0.
	'''
	def init_params(self, d=1):
		self.i_gate = np.random.uniform(-np.sqrt(1. / d), np.sqrt(1. / d), d)
		self.f_gate = np.random.uniform(-np.sqrt(1. / d), np.sqrt(1. / d), d)
		self.o_gate = np.random.uniform(-np.sqrt(1. / d), np.sqrt(1. / d), d)
		self.ci_gate = np.random.uniform(-np.sqrt(1. / d), np.sqrt(1. / d), d)
		self.gates = gates = np.array([self.i_gate, self.f_gate, self.o_gate, self.ci_gate])

	''' Note: LSTM Cell is stateful, the persence of a sample is affected by
	          the last sample. So, we have to forward samples one by one.
	          LSTM Cell is just a neuron, so batchx is just an array not a matrix.
	'''
	def forward(self, batchx):
		self.batchx = batchx
		batchzs = []
		batchas = []
		batchprevs = []
		batchc = []
		batchh = []
		for x in batchx:
			# Update four gates
			zs = np.dot(self.gates, [[x], [self.h], [1]])
			batchzs.append(zs)

			i = self.i_activation.activation(zs[0])
			f = self.f_activation.activation(zs[1])
			o = self.o_activation.activation(zs[2])
			ci = self.ci_activation.activation(zs[3])
			batchas.append([i, f, o, ci])

			batchprevs.append([self.c, self.h])
			# Update state and output
			self.c = f * self.c + i * ci
			self.h = o * self.activation.activation(c)
			batchc.append(self.c)
			batchh.append(self.h)

		self.batchzs = np.array(batchzs)
		self.batchas = np.array(batchas)
		self.batchprevs = np.array(batchprevs)
		self.batchc = np.array(batchc)
		self.batchh = np.array(batchh)

		return self.batchh

	''' bachda is an array not a matrix.
	'''
	def backward(self, batchda):
		batchdz = []
		# The following `*` are Hadamard Production
		i = self.batchas[0]
		f = self.batchas[1]
		o = self.batchas[2]
		ci = self.batchas[3]
		c_1 = self.batchprevs[0]
		c = self.batchc

		# dc = @C/@h * @h/@c = @C/@h * o * tanh'(c)
		dc = batchdz * self.batchas[2] * self.activation.derivative(c)
		# di = @C/@h * @h/@i = @C/@h * @h/@c * @c/@i = dc * ci
		di = dc * ci
		# df = @C/@h * @h/@f = @C/@h * @h/@c * @c/@f = dc * c[t-1]
		df = dc * c_1
		# do = @C/@h * @h/@o = @C/@h * tanh(c)
		do = self.activation.activation(c)
		# dci = @C/@h * @h/@ci = @C/@h * @h/@c * @c/@ci = @C/@h * dc * i
		dci = dc * i

		# dzs =  @C/@h * @h/@z * @z/@(zs)
		# dzs is a batch_size * 4 matrix
		ifoc = np.array([di, df, do, dci]).T
		dzs = ifoc * self.i_activation.derivative(zs)

		# dx = @C/@h * @h/@x = ...
		#    = o * tanh'(c) * ( sig'(zs[F]) * u_F * c[t-1] 
		#                       + i * sig'(zs[CI]) * u_CI 
		#                       + sig'(zs[I]) * u_I * ci )
		#       + sig'(zs[O] * u_O * tanh(c)
		zI = self.batchzs[0]
		zF = self.batchzs[1]
		zO = self.batchzs[2]
		zCI = self.batchzs[3]
		uI = self.i_gate[0]
		uF = self.f_gate[0]
		uO = self.o_gate[0]
		uCI = self.ci_gate[0]
		dx = o * self.activation.derivative(c) 
		       * ( self.f_activation.derivative(zF) * uF * c_1 \
		           + i * self.ci_activation.derivative(CI) * uCI \
		           + self.i_activation.derivative(zI) * uI * ci ) \
		       + self.o_activation.derivative(zO) * uO * self.activation.activation(c)

		return dx

	def update(self, lr=0.005):
		m = len(self.batchz)
		for i in range(m):
			x = self.batchx[i]
			h = self.batchprevs[1][i]
			dz = dzs[i]
			self.gates -= lr / m * np.dot(dz.T, np.array([x, h, 1]))


class LstmLayer(Layer):
	''' LSTM Layer, which is consist of multiple LSTM cells
	'''
	def __init__(self, dim=1):
		self.dim = dim
		self.cells = [LstmCell() for i in range(dim)]
		for cell in cells:
			cell.init_params(dim)

	def init_params(self, input_dim=1):
		super(self, Layer).init_params(input_dim)

	''' batchxS is a dim * batch_size matrix
	    batchzS is a dim * batch_size matrix
	'''
	def forward(self, batchX):
		self.batchX = batchX
		batchxS = np.dot(self.batchX, self,w).T
		batchaS = []
		for i in range(self.dim):
			cell = self.cells[i]
			batchx = self.batchxS[i]
			batcha = cell.forward(batchx)
			batchaS.append(batcha)
		self.batchA = np.array(batchaS).T

		return self.batchA

	'''
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

		# Derivative in LSTM is so different
		daS = batchDa.T
		dzS = []
		for i in range(self.dim):
			cell = self.cells[i]
			dz = cell.backward(daS[i])
			dzS.append(dz)
		batchDz = np.array(dzS).T

		return self.batchDz

	def update(self, lr=0.005):
		for cell in self.cells:
			cell.update(lr)
		m = len(self.batchDz)
		for i in range(m):
			self.w -= lr / m * np.dot(self.batchDz[i], batchX[i].T)