import numpy as np
from copy import deepcopy
''' Some kinds of SGD algorithms
    Our target is to decide how to update theta
'''

class SGD:

    ''' Stochastic Gradient Descent:
            g[t] = ∇[θ[t−1]]f(θ[t−1])
            θ[t] = θ[t-1] - η * g[t]
    '''
    def update(self, g):
        return g

class Momentum(SGD):

    ''' Momentum:
            g[t] =  ∇[θ[t−1]]f(θ[t−1])
            m[t] = µ * m[t-1] + g[t]
            θ[t] = θ[t-1] - η * m[t]
        Super parameters:
            m: an array or a matrix
            µ: a value, recommended value is ?
    '''
    def __init__(self, mu=0.09):
    	self.m = None
    	self.mu = mu

    def update(self, g):
        if self.m is None:
        	# Use 1.0 is for copy and format
        	self.m = 1.0 * g
        else:
        	self.m = self.mu * self.m + g
        
        delta = self.m

        return delta

class NAG(SGD):

    ''' Nesterov Accelerated Gradient:
            It is so different !
            g[t] =  ∇[θ[t−1]]f(θ[t−1] - η * µ * m[t-1])
            m[t] = µ * m[t-1] + g[t]
            θ[t] = θ[t-1] - η * m[t]
            Transfer:
                g[t] = ∇[θ[t−1]]f(θ[t−1])
                m[t] =  µ * m[t-1] + g[t] + µ * (g[t] - g[t-1])
                θ[t] = θ[t-1] - η * m[t]
            Super parameters:
                g_1: the last g
                m: an array or a matrix
                µ: a value, recommended value is ?
    '''
    def __init__(self, mu=0.09):
    	self.g_1 = None
    	self.m = None
    	self.mu = mu

    def update(self, g):
    	if self.g_1 is None:
    		self.g_1 = np.zeros_like(g)
    	if self.m is None:
        	self.m = g + self.mu * (g - self.g_1)
        else:
        	self.m = self.mu * self.m + g + self.mu * (g - self.g_1)
        
        delta = self.m

        return delta

class AdaGrad(SGD):

    ''' Adapative Gradient:
            g[t] = ∇[θ[t−1]]f(θ[t−1])
            n[t] = n[t-1] + g[t]^2
            θ[t] = θ[t-1] - η * g[t] / sqrt(n[t] + ε)
        Super parameters:
            n: an array or a matrix
            ε: a value, recommended value is ?
    '''
    def __init__(self, epsilon=0.001):
    	self.n = None
    	self.epsilon = epsilon

    def update(self, g):
        if self.n is None:
        	self.n = g * g
        else:
        	self.n = self.n + g * g

        delta = self.g / np.sqrt(self.n + self.epsilon)
        delta = np.nan_to_num(delta)

        return delta

class RMSProp(SGD):

    ''' RMSProp:
            g[t] = ∇[θ[t−1]]f(θ[t−1])
            n[t] = v * n[t-1] + (1-v) * g[t]^2
            θ[t] = θ[t-1] - η * g[t] / sqrt(n[t] + ε)
        Super parameters:
            n: an array or a matrix
            v: a valuem, v is a constant value 0.5?
            ε: a value, recommended value is ?
    '''
    def __init__(self, v=0.5, epsilon=0.001):
    	self.n = None
    	self.v = v
    	self.epsilon = epsilon

    def update(self, g):
        if self.n is None:
        	self.n = (1 - self.v) * g * g
        else:
        	self.n = self.v * self.n + (1 - self.v) * g * g

        delta = self.g / np.sqrt(self.n + self.epsilon)
        delta = np.nan_to_num(delta)

        return delta

class AdaDelta(SGD):

    ''' Adapative Delta:
            g[t] = ∇[θ[t−1]]f(θ[t−1])
            n[t] = v * n[t-1] + (1-v) * g[t]^2
            θ[t] = θ[t-1] - η * g[t] / sqrt(n[t] + ε)
        Super parameters:
            n: an array or a matrix
            v: a valuem, recommended value is 
            ε: a value, recommended value is ?
    '''
    def __init__(self, v=0.1, epsilon=0.001):
    	self.n = None
    	self.v = v
    	self.epsilon = epsilon

    def update(self, g):
        if self.n is None:
        	self.n = (1 - self.v) * g * g
        else:
        	self.n = self.v * self.n + (1 - self.v) * g * g

        delta = self.g / np.sqrt(self.n + self.epsilon)
        delta = np.nan_to_num(delta)

        return delta

class Adam(SGD):

	''' Adam:
	        g[t] = ∇[θ[t−1]]f(θ[t−1])
	        m[t] = µ * m[t-1] + (1-µ) * g[t]
	        ^m[t] = m[t] / (1-µ^t)
	        n[t] = v * n[t-1] + (1-v) * g[t]^2
	        ^n[t] = n[t] / (1-v^t)
	        θ[t] = θ[t-1] - η * ^m[t] / sqrt(^n[t] + ε) 
	    Super parameters:
	        m: an array or a matrix
	        n: an array or a matrix
	        µ: a value, recommended value is ?
	        v: a value, recommended value is ?
	        ε: a value, recommended value is ?
	'''
	def __init__(self, mu, v, epsilon):
		self.m = None
		self.n = None
		self.mu = mu
		self.v = v
		self.epsilon = epsilon
		self.mut = 1
		self.vt = 1
	
	def update(self, g):
		if self.m is None:
			self.m = (1 - self.mu) * g
			self.n = (1 - self.v) * g
		else:
			self.m = self.mu * self.m + (1 - self.mu) * g
			self.v = self.v * self.n + (1 - self.v) * g * g

		self.mut *= self.m 
		self.vt *= self.v 

		_m = self.m / (1 - self.mut)
		_n = self.n / (1 - self.vt)

		delta = _m / (np.sqrt(_n) + self.epsilon)
		delta = np.nan_to_num(delta)

		return delta

class NAGR(SGD):

	''' NAGR:
	        g[t] = ∇[θ[t−1]]f(θ[t−1])
	        m[t] = µs[t] * m[t-1] + g[t]
	        ^m[t] = g[t] +  µs[t+1] * m[t]
	        θ[t] = θ[t-1] - η * ^m[t]
	    Super parameters:
	        m: an array or a matrix
	        µs: an array generated for µ
	        µ: a value, recommended value is ?
	'''
	def __init__(self, mu=0.99):
		self.m = None
		self.mu = mu
		self.e = np.pow(0.96, 1.0/250)
		self.et = 0.5
		self.mu_t =  mu * 0.5

	def update(self, g):
		self.et *= self.e
		self.mu_t = self.mu * (1 - self.et)
		if self.m is None:
			self.m = 1.0 * g
		else:
			self.m = self.mu_t * self.m + g

		mu_t_next = self.mu * (1 - self.et * self.e)
		_m = g + mut_next * self.m

		delta = _m

		return delta


class Nadam:

	''' Nadam:
	        g[t] = ∇[θ[t−1]]f(θ[t−1])
	        ^g = g[t] / (1 - Mul(i=1->t)µs[i])
	        m[t] = µ * m[t-1] + (1-µ) * g[t]
	        ^m[t] = m[t] / (1 - Mul(i=1->t+1)µs[i])
	        n[t] = v * n[t-1] + (1-v) * g[t]^2
	        ^n[t] = n[t] / (1 - v^t)
	        _m[t] = (1 - µs[t]) * ^g[t] + µs[t+1] * ^m[t]
	        θ[t] = θ[t-1] - η * _m[t] / sqrt(^n[t] + ε)
	    Super parameters:
	        m: an array or a matrix
	        n: an array or a matrix
	        µ: a value, recommended value is 0.99?
	        µs: an array generated from µ
	        v: a value, recommended value is 0.999?
	        ε: a value, recommended value is 1e-8?
	'''
	def __init__(self, mu=0.99, v=0.999, epsilon=1e-8):
		self.m = None
		self.n = None
		self.mu = mu
		self.e = np.pow(0.96, 1.0/250)
		self.et = 0.5
		self.mu_t = mu * 0.5
		self.mum_t = 1
		self.v = v
		self.vt = 1
		self.epsilon = epsilon

	def update(self, g):
		self.et *= self.e
		self.mu_t = self.mu * (1 - self.et)
		self.mum_t *= self.mu_t
		
		_g = g / (1 - self.mum_t)
		self.m = self.mu * m + (1 - self.mu) * g
		
		mu_t_next = self.mu * (1 - self.et * self.e)
		mum_t_next = self.mum_t * mu_t_next
		_m = m / ((1 - mum_t_next))
		
		self.vt *= self.v
		n = self.n / (1 - self.vt)
		__m = (1 - self.mu_t) * _g + mu_t_next * _m

		delta = __m / np.sqrt(_n + epsilon)
		delta = np.nan_to_num(delta)

		return delta


sgd = SGD()
momentum = Momentum()
nag = NAG()
adagrad = AdaGrad()
rmsprop = RMSProp()
adadelta = AdaDelta()
adam = Adam()
nagr = NAGR()
nadam = Nadam()


