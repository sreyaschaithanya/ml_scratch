import numpy as np

class Layers:
    """
    Base class for layers
    """
    def __init__(self):
        self.params = {}
    
    def forward(self,input):
        raise NotImplementedError
    
    def backward(self,grad):
        raise NotImplementedError

class Dense(Layers):
    def __init__(self, input_size, neurons):
        """
        Make a params dictionary and store weights and bias
        """
        super().__init__()
        self.input_size  = input_size
        self.neurons = neurons
        self.params["w"] = np.random.randn(input_size,neurons)
        self.params["b"] = np.random.randn(neurons)

    def forward(self, input):
        """
        Linear Output
        """
        self.input = input
        return input @ self.params["w"] + self.params["b"]

    def backward(self,grad):
        """
        Bacpropogation for linear layer
        inflow gradient * local gradient
        local gradient = input
        """
        return np.dot(self.input.T , grad)

def Activation(Layers):
    """
    Base class for all activation functions
    Will take input and compute the output
    """
    def __init__(self):
        pass
    
    def forward(self,input):
        pass

    def backward(self, grad):
        pass