import numpy as np
import copy

def handle_zeros(scale):
	if np.isscalar(scale):
		if scale == .0:
			scale = 1.
	elif isinstance(scale, np.ndarray):
		scale = scale.copy()
		scale[scale == 0.0] = 1.0
	return scale

class MinMaxScaler:
	def __init__(self, feature_range=(0,1), max_refer=None, min_refer=None):
		assert(feature_range[0] < feature_range[1])
		self.feature_range = feature_range
		self.max_refer = max_refer
		self.min_refer = min_refer

	def transform(self, X):
		data_min = np.min(X, axis=0)
		data_max = np.max(X, axis=0)

		if self.max_refer is not None:
			data_max = self.max_refer(data_max)

		if self.min_refer is not None:
			data_min = self.min_refer(data_min)

		data_range = data_max - data_min
		self.scale = 1.0 * (self.feature_range[1] - self.feature_range[0]) / handle_zeros(data_range)
		self.min = self.feature_range[0] - data_min * self.scale
		self.data_min = data_min
		self.data_max = data_max

		X2 = copy.deepcopy(X)
		X2 *= self.scale
		X2 += self.min

		return X2

	def inverse(self, X):
		X2 = copy.deepcopy(X)
		X2 -= self.min
		X2 /= self.scale

		return X2