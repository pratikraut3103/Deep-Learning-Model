from Layers import Base
import copy
from Optimization import Constraints
class NeuralNetwork():
    def __init__(self,optimizer,weights_initializer,bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer=weights_initializer
        self.bias_initializer=bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer =None
        self.loss_layer=None
        self.phase = False

    def forward(self):
        reg_loss = 0
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)
            if layer.trainable:
                if layer.optimizer:
                    if layer.optimizer.regularizer:
                        reg_loss += layer.optimizer.regularizer.norm(layer.weights)

        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor) + reg_loss
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self,layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self,iterations):
        for layer in self.layers:
            layer.testing_phase = False
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()



    def test(self,input_tensor):
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    @property
    def phase(self):
        return self.__phase

    @phase.setter
    def phase(self, phase):
        self.__phase = phase