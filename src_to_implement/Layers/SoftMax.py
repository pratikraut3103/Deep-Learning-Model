import numpy as np
from Layers import Base
class SoftMax(Base.BaseLayer) :
    def __int__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor = input_tensor - np.max(input_tensor)
        n= np.exp(input_tensor)
        self.output = n / np.sum(n, axis=1, keepdims=True)
        return self.output

    def backward(self,error_tensor):
         self.error_tensor = error_tensor
         return self.output * (self.error_tensor - (self.output * self.error_tensor).sum(axis=1)[:,None])