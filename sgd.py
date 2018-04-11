
''' Some kinds of SGD algorithms
    Our target is to decide how to update theta
'''

class SGD:

	''' Stochastic Gradient Descent:
	        g[t] = J'(theta[t-1])
	        theta[t] = theta[t-1] - eta * g[t]
	'''
	def update(self, theta, g, eta):
		return theta - eta * g

class Momentum:

	''' Momentum:
	        g[t] = J'(theta[t-1])
	        v[t] = gamma * v[t-1] + eta * g[t]
	        theta[t] = theta[t-1] - v[t]
	'''
	def update(self, theta, g, eta, v, gamma):
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
	''' Adam:
	        g[t] = J'(theta[t-1])
	        The last `*` represents Hadamard Production here !
	        m[t] = beta1 * m[t-1] + (1-beta1) * g[t]
	        The last `*` represents Hadamard Production here !
	        G[t] = gamma * G[t] + (1-gamma) * g[t] * g[t]
	        alpha = eta * sqrt(1-gamma^t) / (1-beta^t)
	        theta[t] = theta[t-1] - alpha * m[t] / sqrt(G[t] + epsilon) 
	'''

sgd = SGD()
momentum = Momentum()
nag = NAG()
adagrad = AdaGrad()
rmsprop = RMSProp()
adadelta = AdaDelta()
adam = Adam()