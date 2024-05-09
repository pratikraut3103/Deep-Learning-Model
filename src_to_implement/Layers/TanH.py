from Layers import Base
import numpy as np


class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input_tensor):
        '''
        returns the input_tensor for the next layer.
        '''
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        '''
        returns the error_tensor for the next layer.
        '''
        return error_tensor * (1 - self.activation **2)

