from activation import *
from layer import *
import numpy as np


class LstmLayer(Layer):
	''' LSTM Layer, which is consist of dim LSTM cells.
		There are four types gates in LSTM:
	        InputGate:     i =  sig (ui * x + wi * h + bi)
	        ForgetGate:    f =  sig (uf * x + wf * h + bf)
	        OutputGate:    o =  sig (uo * x + wo * h + bo)
	        GXXGate:       g = tanh(uc * x + wc * h + bc)
	    Update state and output:
	    	State:  c = f * c + i * g
	    	Output: h = o * tanh(c)

	    So, there are dim (i, f, o, g) gates, and each 
	    gate has (u,w,b) properties 
	'''
	def __init__(self, dim=1):
		self.dim = dim
		# All following properties are arrays whose length is dim
		# i gates
		self.ui = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		self.wi = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		self.bi = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		# f gates
		self.uf = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		self.wf = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		self.bf = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		# o gates of dim
		self.uo = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		self.wo = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		self.bo = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		# g gates of dim cells
		self.ug = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		self.wg = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		self.bg = np.random.uniform(-np.sqrt(1. / dim), np.sqrt(1. / dim), dim)
		# c
		self.c = np.zeros(dim)
		# h
		self.h = np.zeros(dim)

		# Set activations
		self.iactivation = sig
		self.factivation = sig
		self.oactivation = sig
		self.gactivation = tanh
		self.activation = tanh

	def init_params(self, input_dim=1):
		super(self, Layer).init_params(input_dim)

	''' Forward progation
	    batchX is a batch_size * input_dim matrix
	    self.w is a dim * input_dim matrix
	    self.batchZ is a batch_size * dim matrix

	    Note: LSTM Cell is stateful, the persence of a sample is affected by
	          the last sample. So, we have to forward samples one by one.
	'''
	def forward(self, batchX):
		self.batchX = batchX
		self.batchZ = np.dot(batchX, self,w)
		batchA = []

		# All following varaibles are batch_size * dim matrix
		# Record tmp results for backward propagation
		batchzi = []
		batchzf = []
		batchzo = []
		batchzg = []
		batchi = []
		batchf = []
		batcho = []
		batchg = []
		batchc_1 = []
		batchh_1 = []
		batchc = []
		batchac = []
		batchh = []
		# Forward self.batchZ to LSTM cells as batchX
		for x in self.batchZ:
			# The following `*` are Hadamard Productions
			# Calculate weighted sums
			zi = self.ui * x + self.wi * self.h + self.bi
			zf = self.uf * x + self.wf * self.h + self.bf
			zo = self.uo * x + self.wo * self.h + self.bo
			zg = self.ug * x + self.wg * self.h + self.bg
			batchzi.append(zi)
			batchzf.append(zf)
			batchzo.append(zo)
			batchzg.append(zg)

			# Calculate activation values
			i = self.iactivation.activation(zi)
			f = self.factivation.activation(zf)
			o = self.oactivation.activation(zo)
			g = self.gactivation.activation(zg)
			batchi.append(i)
			batchf.append(f)
			batcho.append(o)
			batchg.append(g)

			# Update state and output
			batchc_1.append(self.c)
			batchh_1.append(self.h)
			self.c = f * self.c + i * ci
			# Calcualte activation value of c
			ac = self.activation.activation(c)
			self.h = o * ac
			batchc.append(self.c)
			batchac.append(self.ac)
			batchh.append(self.h)

			batchA.append(self.h)

		self.batchA = np.array(batchA)

		self.batchzi = np.array(batchzi)
		self.batchzf = np.array(batchzf)
		self.batchzo = np.array(batchzo)
		self.batchzg = np.array(batchzg)
		self.batchi = np.array(batchi)
		self.batchf = np.array(batchf)
		self.batcho = np.array(batcho)
		self.batchg = np.array(batchg)
		self.batchc_1 = np.array(batchc_1)
		self.batchh_1 = np.array(batchh_1)
		self.batchc = np.array(batchc)
		self.batchac = np.array(batchac)
		self.batchh = np.array(batchh)

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
		# All following `*` are Hadamard Productions
		# dc = @C/@h * @h/@c = da * o * tanh'(c)
		batchDc = batchDa * self.batcho * self.activation.derivative(self.batchc)
		# di = @C/@c * @c/@i = dc * g
		batchDi = batchDc * self.batchg
		# df = @C/@c * @c/@f = dc * c_1
		batchDf = batchDc * self.batchc_1
		# do = @C/@h * @h/@o = da * ac
		batchDo = batchDa * self.batchac
		# dg = @C/@c * @c/@g = dc * i
		batchDg = batchDc * self.batchi

		# dzi = @C/@i * @i/@zi = di * sig'(zi)
		batchDzi = batchDi * self.iactivation.derivative(self.batchzi)
		# dzf = @C/@f * @f/@zf = df * sig'(zf)
		batchDzf = batchDf * self.iactivation.derivative(self.batchzf)
		# dzo = @C/@o * @o/@zo = do * sig'(zo)
		batchDzo = batchDo * self.iactivation.derivative(self.batchzo)
		# dzg = @C/@g * @g/@zg = dg * tanh'(zg)
		batchDzg = batchDg * self.iactivation.derivative(self.batchzg)

		# dx = @C/@h * @h/@x = da * (o * tanh'(c) * (sig'(zf) * uf * c_1 + 
		#                                            i * sig'(zg) * ug + 
		#                                            sig'(zi) * ui * g
		#                                            ) + 
		#                            sig'(zo) * uo * ac
		#                           )
		m = len(batchDa)
		batchuf = np.array([self.batchuf] * m)
		batchug = np.array([self.batchug] * m)
		batchui = np.array([self.batchui] * m)
		batchDx = batchDa * ( self.batcho * self.activation.derivative(self.batchc) \
			                   * (self.factivation.derivative(self.batchzf) * batchuf * self.batchc_1 \
			                   	   + self.batchi * self.gactivation.derivative(self.batchzg) * batchug \
			                   	   + self.iactivation.derivative(self.batchzi) * batchui * self.batchg) 
			                   + self.oactivation.derivative(self.batchzo) * batchuo * self.batchac)

		self.batchDz = batchDx

		self.batchDzi = batchDzi
		self.batchDzf = batchDzf
		self.batchDzo = batchDzo
		self.batchDzg = batchDzg

		return self.batchDz

	def update(self, lr=0.005):
		m = len(self.batchDz)
		for i in range(m):
			# Upate self.ui, self.wi, self.bi
			self.ui -= lr / m * self.batchDzi[i] * self.batchZ[i]
			self.wi -= lr / m * self.batchDzi[i] * self.batchh_1[i]
			self.bi -= lr / m * self.batchDzi[i]
			# Upate self.uf, self.wf, self.bf
			self.uf -= lr / m * self.batchDzf[i] * self.batchZ[i]
			self.wf -= lr / m * self.batchDzf[i] * self.batchh_1[i]
			self.bf -= lr / m * self.batchDzf[i]
			# Upate self.uo, self.wo, self.bo
			self.uo -= lr / m * self.batchDzi[i] * self.batchZ[i]
			self.wo -= lr / m * self.batchDzi[i] * self.batchh_1[i]
			self.bo -= lr / m * self.batchDzi[i]
			# Upate self.ug, self.wg, self.bg
			self.ug -= lr / m * self.batchDzi[i] * self.batchZ[i]
			self.wg -= lr / m * self.batchDzi[i] * self.batchh_1[i]
			self.bo -= lr / m * self.batchDzi[i]

			# Upate self.w
			self.w -= lr / m * np.dot(self.batchDz[i], batchX[i].T)