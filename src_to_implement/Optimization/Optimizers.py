#Implementing Optimizers
import numpy as np


class base_optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(base_optimizer):
    def __init__(self,learning_rate):
        super().__init__()
        self.learning_rate=learning_rate
    #Returning the updated weights
    def calculate_update(self,weight_tensor, gradient_tensor):
           # return weight_tensor - self.learning_rate*gradient_tensor
        if self.regularizer:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * shrinkage

        return weight_tensor - self.learning_rate*gradient_tensor



class SgdWithMomentum(base_optimizer):
    def __init__(self, learning_rate,momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v=0

    def calculate_update(self,weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        # return weight_tensor + self.v

        if self.regularizer:
           shrinkage = self.regularizer.calculate_gradient(weight_tensor)
           weight_tensor = weight_tensor - self.learning_rate * shrinkage

        return weight_tensor + self.v


class Adam(base_optimizer):
    def __init__(self, learning_rate,mu,rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho =rho
        self.v=0
        self.r=0
        self.eps = 1e-8
        self.t =1

    def calculate_update(self,weight_tensor, gradient_tensor):
        self.v = self.mu * self.v  +(1-self.mu) * gradient_tensor
        self.r= self.rho * self.r + (1-self.rho)* (gradient_tensor ** 2)
        v_corrected = self.v / (1-self.mu ** self.t)
        r_corrected = self.r /(1-self.rho ** self.t)
        self.t =self.t+1

        if self.regularizer:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * shrinkage
        weight_tensor = weight_tensor - (self.learning_rate * v_corrected) / (
                        np.sqrt(r_corrected) + np.finfo(float).eps)

        return weight_tensor