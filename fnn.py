from loss import *
from sgd import *
import numpy as np

losses = {'mse': mse, 'ce': ce}

optimizers = {'sgd': sgd, "momentum": momentum}

class FNN:
    ''' Full-connection Neural Network
        It is a simple network with multiple layers
    '''
    def __init__(self):
        self.layers = []

    ''' Forward-progation
        batchX is a batch_size * M matrix
        self.batchA a batch_size * N matrix
    '''
    def forward(self, batchX):
        batchA = batchX
        for layer in self.layers:
            batchA = layer.forward(batchA)

        self.batchA = self.output.forward(batchA)

        return self.batchA

    # Backward-progation
    def backward(self, delta_loss):
        batchDz = self.output.backward(delta_loss)
        
        last_layer = self.output
        layer_size = len(self.layers)
        for k in range(layer_size):
            i = layer_size - 1 - k
            layer = self.layers[i]
            batchDz = layer.backward(batchDz, last_layer.w)
            last_layer = layer

    def update(self, lr=0.005):
        self.output.update(lr)
        layer_size = len(self.layers)
        for k in range(layer_size):
            i = layer_size - 1 - k
            layer = self.layers[i]
            layer.update(lr)

    # Add a hidden layer
    def add(self, layer):
        self.layers.append(layer)

    # Set output layer
    def output(self, layer):
        self.output = layer

    def compile(self, loss='mse', optimizer='sgd'):
        self.loss = losses[loss]
        self.optimizer = optimizers[optimizer]


    def fit(self, trainX, trainY, \
        lr=0.005, batch_size=1, epochs=30, shuffle=False, verbose=2):
        assert(len(trainX) == len(trainY))
        # self.trainX = trainX
        # self.trainY = trainY

        # Init layers' params
        self.dim = len(trainX[0])
        input_dim = self.dim
        for layer in self.layers:
            layer.init_params(input_dim)
            input_dim = layer.dim
        self.output.init_params(input_dim)

        # Start train
        size = len(trainX)
        batchs = size / batch_size
        if size % batch_size != 0:
            batchs += 1

        batch_pool = None
        if epochs > 1:
            batch_pool = []
        for i in range(epochs):
            
            for j in range(batchs):
                # Select a batch of samples
                if batch_pool == None or i == 0:
                    start = j * batch_size
                    batchX = np.array(trainX[start:start + batch_size])
                    batchY = np.array(trainY[start:start + batch_size])
                    if batch_pool != None:
                        batch_pool.append((batchX, batchY))
                else:
                    (batchX, batchY) = batch_pool[j] 
                    
                # Forward progation
                self.forward(batchX)
                # Backward progation
                loss = self.loss.loss(self.batchA, batchY)
                if verbose < 3:
                    print "[loss: ", loss, "]"
                delta_loss = self.loss.delta(self.batchA, batchY)
                self.backward(delta_loss)
                # Update
                self.update(lr)


    def predict(self, X):
        Y = self.forward(X)

        return Y