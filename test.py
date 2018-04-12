from fnn import FNN
from layer import Layer
from lstm import LstmLayer

fnn = FNN()
fnn.add(LstmLayer(4))
fnn.output(Layer(1))
fnn.compile()
train_x = [[0.1,0.2,0.3],[0.2,0.3,0.4],[0.3,0.4,0.5],[0.4,0.5,0.6],[0.5,0.6,0.7],[0.6,0.7,0.8]]
train_y = [[0.4],[0.5],[0.6],[0.7],[0.8],[0.9]]
fnn.fit(train_x, train_y, batch_size=6, epochs=3000, lr=0.5)
test_x = [[0.7,0.8,0.9]]
test_y = [[1.0]]
pred_y = fnn.predict(test_x)
print "test_y:", test_y
print "pred_y:", pred_y