from fnn import FNN
from layer import Layer
from lstm import LstmLayer

fnn = FNN()
fnn.add(LstmLayer(4))
fnn.output(Layer(1))
fnn.compile()
train_x = [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]]
train_y = [[4],[5],[6],[7],[8],[9]]
fnn.fit(train_x, train_y, batch_size=6, epochs=100, lr=0.5)
test_x = [[8,9,10]]
test_y = [[11]]
pred_y = fnn.predict(test_x)
print "test_y:", test_y
print "pred_y:", pred_y