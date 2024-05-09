#Implementing Base Layer which will be inherited by every layer
class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.testing_phase = False