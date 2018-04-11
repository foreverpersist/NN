
''' Some kinds of Loss Functions
'''

class MSE:
	''' Mean Sqaure Error:
	        C = sum((y-a)^2)
	'''
	def loss(self, batchA, batchY):
		batchD = batchA - batchY
		return np.sqaure(batchD)

	''' Delta: 
	        @C/@aL
	'''
	def delta(self, batchA, batchY):
		return 2 * (batchA - batchY)


class CE:
	''' Cross Entropy:
	        C = yloga + (1-y)log(1-a)
	'''
	def loss(self, batchA, batchY):
		return

	def delta(self, batchA, batchY):
		return

mse = MSE()
ce = CE()