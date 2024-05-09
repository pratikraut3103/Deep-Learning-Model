from Layers import Base
import numpy as np


class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input_tensor):
        '''
        returns the input_tensor for the next layer.
        '''
        self.activation = 1 / (1 + np.exp(-1 * input_tensor))
        return self.activation

    def backward(self, error_tensor):
        '''
        returns the error_tensor for the next layer.
        '''
        return error_tensor * self.activation * (1 - self.activation)