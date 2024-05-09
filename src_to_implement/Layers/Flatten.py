import numpy as np
from Layers import Base

class Flatten(Base.BaseLayer):

    def _init_(self):
        super().__init__()
        self.shape = 0

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self,error_tensor):
        error_tensor = error_tensor.reshape(self.shape)
        return error_tensor