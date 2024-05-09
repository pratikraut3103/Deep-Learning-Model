import numpy as np
from Layers import Base

import copy


class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.output_size = output_size

        self.__weights = np.random.uniform(0, 1, (self.input_size, self.output_size))

        self.__optimizer = None
        self.optimizerWeights = None
        self.bias = np.ones(self.output_size)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        result = np.dot(self.input_tensor, self.__weights)
        result = result + self.bias

        return result

    def backward(self, error_tensor):
        error_new = np.dot(error_tensor, self.__weights.transpose())
        self.__gradient_weights = np.dot(self.input_tensor.transpose(), error_tensor)
        self.__gradient_bias = error_tensor.mean(axis=0) * self.input_tensor.shape[0]

        if self.__optimizer != None:
            self.weights = self.optimizerWeigths.calculate_update(self.weights, self.__gradient_weights)
            self.bias = self.optimizerBias.calculate_update(self.bias, self.__gradient_bias)
        return error_new

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.__gradient_weights = gradient_weights

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        self.__weights = weights

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
        self.optimizerWeigths = copy.deepcopy(self.__optimizer)

        self.optimizerBias = copy.deepcopy(self.__optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)






