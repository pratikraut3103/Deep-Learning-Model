import numpy as np

class CrossEntropyLoss() :
    def __init__(self):
        pass
    def forward(self,prediction_tensor, label_tensor):
        self.label_tensor = label_tensor
        self.prediction_tensor = prediction_tensor
        return np.sum(label_tensor * -np.log(prediction_tensor + np.finfo(float).eps))





    def backward(self,label_tensor):
        self.label_tensor = label_tensor
        return np.where(self.label_tensor == 1, -1 / self.prediction_tensor, 0)

