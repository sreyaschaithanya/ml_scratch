import numpy as np

class Loss():
    """
    This is the abstract class for the loss functions
    """
    def loss(self, predicted, actual):
        raise NotImplementedError
    
    def grad(self,predicted,actual):
        raise NotImplementedError

class MSE(Loss):
    """
    This is the mean square error loss function
    """
    def loss(self, predicted, actual):
        loss = np.sum((predicted - actual)**2)
        return  loss/predicted.shape[1]

    
    def grad(self, predicted, actual):
        return 2 * (predicted - actual)/predicted.shape[1]       