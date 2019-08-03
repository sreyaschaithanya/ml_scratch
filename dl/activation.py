import numpy as np
from autograd import grad as gradient

from dl.layers import Layers

class Activation(Layers):
    """
    Base class for all activation functions
    Will take input and compute the output
    """
    def __init__(self,f):
        self.f = f
        self.f_dash = gradient(f)

    def forward(self,input):
        self.input = input
        return self.f(input)

    def backward(self, grad):
        return self.f_dash(self.input) * grad

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh)

def tanh(x):
    return(np.tanh(x))

def logistic(z):
    return (1/(1+np.exp(-z)))
