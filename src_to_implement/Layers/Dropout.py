import numpy as np
from Layers import Base


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.testing_phase = False
        self.v = 0

    def forward(self, input_tensor):
        if not self.testing_phase:
            self.v = np.random.binomial(1,self.probability, size=input_tensor.shape)
            input_tensor = np.multiply(input_tensor, self.v)
            input_tensor /= self.probability
        return input_tensor

    def backward(self, error_tensor):

        error_tensor = np.multiply(error_tensor, self.v/self.probability)
        return error_tensor
