import numpy as np
import math
class L2_Regularizer:
    def __init__(self,alpha):
        self.alpha = alpha

    def calculate_gradient(self,weights):
        return self.alpha * weights

    def norm(self,weights):
        sum = np.sum(np.square(weights))
        return self.alpha*sum

class L1_Regularizer:
    def __init__(self,alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self,weights):
        sum = np.sum(np.abs(weights))
        return self.alpha*sum




