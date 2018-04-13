from fnn import FNN
from layer import Layer
from lstm import LstmLayer
from preprocess import MinMaxScaler

def max_refer(max):
	return 2 * max

fnn = FNN(3)
fnn.add(LstmLayer(4))
fnn.add(Layer(1))
fnn.compile()
x = [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8], [7,8,9]]
y = [[4],[5],[6],[7],[8],[9], [10]]
scaler = MinMaxScaler(max_refer=max_refer)
data_x = scaler.transform(x)
data_y = scaler.transform(y)
train_x = data_x[:-1]
train_y = data_y[:-1]
fnn.fit(train_x, train_y, batch_size=6, epochs=3000, lr=0.5)

test_x = [data_x[-1]]
test_y = [y[-1]]
pred_y = fnn.predict(test_x)
pred_y = scaler.inverse(pred_y)
print "test_y:", test_y
print "pred_y:", pred_y
