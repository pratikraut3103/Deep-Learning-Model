import numpy as np
from Layers import Base

class ReLU (Base.BaseLayer) :
    def __init__(self):
        super().__init__()
        #print(self.trainable)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(self.input_tensor>=0, self.input_tensor,0)

    def backward(self, error_tensor):
        return np.where(self.input_tensor<=0,0,error_tensor)