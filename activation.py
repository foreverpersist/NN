import numpy as np

''' Activation Functions
    Input: batchX, maybe is
                            a elemen, such as a float or a int
                            a array whose length is batch_size
                            a matrix whose shape is batch_size * dim
'''

class Sigmoid:

    def primitive(self, batchX):
        return 1.0 / (1.0 + np.exp(-batchX))

    def derivative(self, batchX, batchY=None):
    	if batchY == None:
    		batchY = self.primitive(batchX)
        return (1.0 - batchY) * batchY

class Tanh:

    def primitive(self, batchX):
        return np.tanh(batchX)

    def derivative(self, batchX, batchY=None):
    	if batchY == None:
    		batchY = self.primitive(batchX)
        return (1.0 - np.square(batchY))

class Relu:

    def __primitive(self, x):
        return max(0.0, x)

    def primitive(self, batchX):
        if type(batchX) == np.array:
            res = []
            for x in xlist:
                res.append(__primitive(x))
            return np.array(res)
        return __primitive(xlist)

    def __derivative(self, x):
        if x > 0:
            return 1.0
        return 0.0

    def derivative(self, batchX, batchY=None):
        if type(xlist) == np.array:
            res = []
            for x in xlist:
                res.append(__derivative(x))
            return np.array(res)
        return __derivative(xlist)
        

class Softmax:
    ''' This is a special activation. 
        aL[j] = e^(zL[j]) / sum( e^(zL[k]) )
        Often used in output layer with loss: C = - sum( y[k] * log(a[k]) )
        Derivative:
            j = i:  @a[j]/@z[i] = a[j] * (1-a[]j)
            j != i: @a[j]/@z[i] = a[j] * a[i]
    '''

    # Note xlist is a list or an array
    def primitive(self, xlists):
        elist = np.exp(xlists)
        s = np.sum(elist)
        return 1.0 * elist / s

    def derivative(self, xlists, j, i):
        elist = np.exp(xlists)
        s = np.sum(elist)
        aj = 1.0 * elist[j] / s 
        if j == i:
            return aj * (1.0 - aj)
        else:
            ai = 1.0 * elist[i] / s
            return -1.0 * aj * ai


sig = Sigmoid()
tanh = Tanh()
relu = Relu()
softmax = Softmax()