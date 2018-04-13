import numpy as np
from copy import deepcopy
''' Some kinds of SGD algorithms
    Our target is to decide how to update theta
'''

class SGD:

    ''' Stochastic Gradient Descent:
            g[t] = J'(theta[t-1])
            theta[t] = theta[t-1] - eta * g[t]
    '''
    def update(self, g):
        return g

class Momentum:

    ''' Momentum:
            g[t] = J'(theta[t-1])
            v[t] = gamma * v[t-1] + eta * g[t]
            theta[t] = theta[t-1] - v[t]
    '''
    def update(self, theta, g, eta, others=(0.005, 0.005)):
        (v,gamma) = others
        v = gamma * v + eta * g
        return theta - v

class NAG:

    ''' Nesterov Accelerated Gradient:
            It is so different !
            g[t] = J'(theta[t-1] - gamma * v[t-1])
            v[t] = gamma * v[t-1] + eta * g[t]
            theta[t] = theta[t-1] - v[t]
    '''
    def update(self, theta, g, eta, v, gamma):
        # gt = ?
        v = gamma * v + eta * g
        return theta - v

class AdaGrad:

    ''' Adapative Gradient:
            g[t] = J'(theta[t-1])
            The `*` represents Hadamard Production here !
            G[t] = G[t] + g[t] * g[t]
            The `*` represents Hadamard Production here !
            theta[t] = theta[t-1] - eta / sqrt(G[t] + epsilon) * g[t]
    '''
    def update(self, theta, g, eta, v, gamma):
        pass

class RMSProp:

    ''' RMSProp:
            g[t] = J'(theta[t-1])
            The last `*` represents Hadamard Production here !
            G[t] = gamma * G[t] + (1-gamma) * g[t] * g[t]
            The `*` represents Hadamard Production here !
            theta[t] = theta[t-1] - eta / sqrt(G[t]+epsilon) * g[t]
    '''
    def update(self):
        pass

class AdaDelta:

    ''' AdaDelta:
            g[t] = J'(theta[t-1])
            The last `*` represents Hadamard Production here !
            G[t] = gamma * G[t] + (1 - gamma) * g[t] * g[t]
            The `*` represents Hadamard Production here !
            Dtheta[t] = - sqrt(delta[t-1] + epsilon)/sqrt(G[t] + epsilon) * g[t] 
            theta[t] = theta[t-1] + Dtheta[t]
            The last `*` represents Hadamard Production here !
            delta[t] = gamma * delta[t-1] + (1 - gamma) * Dtheta[t] * Detheta[t]
    '''
    def update(self):
        pass

class Adam:
<<<<<<< Updated upstream:optimizer.py
    ''' Adam:
            g[t] = J'(theta[t-1])
            m[t] = u * m[t-1] + (1-u) * g[t]
            n[t] = v * n[t-1] + (1-v) * g[t]^2
            m~[t] = m[t] / (1-u^t)
            n~[t] = n[t] / (1-v^t)
            theta -= eta * m~[t] / (n~[t] + epsilon)
            The last `*` represents Hadamard Production here !
            m[t] = beta1 * m[t-1] + (1-beta1) * g[t]
            The last `*` represents Hadamard Production here !
            G[t] = gamma * G[t] + (1-gamma) * g[t] * g[t]
            alpha = eta * sqrt(1-gamma^t) / (1-beta^t)
            theta[t] = theta[t-1] - alpha * m[t] / sqrt(G[t] + epsilon) 
    '''
    def update(self, theta, g, eta, others):
        pass
=======
	''' Adam:
	        g[t] = J'(theta[t-1])
	        The last `*` represents Hadamard Production here !
	        m[t] = beta1 * m[t-1] + (1-beta1) * g[t]
	        The last `*` represents Hadamard Production here !
	        G[t] = gamma * G[t] + (1-gamma) * g[t] * g[t]
	        alpha = eta * sqrt(1-gamma^t) / (1-beta^t)
	        theta[t] = theta[t-1] - alpha * m[t] / sqrt(G[t] + epsilon) 
	'''
	def __init__(self, beta1, beta2, epsilon):
		self.m = None
		self.v = None
		self.beta1_t = 1
		self.beta2_t = 1
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
	
	def update(self, g):
		print "g", g

		self.beta1_t = self.beta1_t * self.beta1
		self.beta2_t = self.beta2_t * self.beta2

		if self.m == None:
			self.m = (1-self.beta1) * g
			self.v = (1-self.beta2) * g
		else:
			self.m = self.beta1 * self.m + (1-self.beta1) * g
			self.v = self.beta2 * self.v + (1-self.beta2) * g

		print "m: ", self.m
		print "v: ", self.v
		m_t = self.m / (1-self.beta1_t)
		v_t = self.v / (1-self.beta2_t)

		delta_theta = m_t / (np.power(v_t, 0.5) + self.epsilon)
		delta_theta = np.nan_to_num(delta_theta)
		print "delta_theta: ", delta_theta
		return delta_theta

sgd = SGD()
momentum = Momentum()
nag = NAG()
adagrad = AdaGrad()
rmsprop = RMSProp()
adadelta = AdaDelta()
# adam = Adam(0.9, 0.999, 10e-8, 0.1, 3)
# result = adam.update([1, 1, 1])
# print(adam.beta1_t)
# print(adam.beta2_t)
# print(result)
# result = adam.update([1, 2, 3])
# print(adam.beta1_t)
# print(adam.beta2_t)
# print(result)
>>>>>>> Stashed changes:sgd.py
