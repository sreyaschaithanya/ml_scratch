import numpy as np
from autograd import grad

from ml_scratch.dl.layers import Layers


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - pow(np.tanh(x), 2)


def logistic(z):
    return 1 / (1 + np.exp(-z))


class Activation(Layers):
    """
    Base class for all activation functions
    Will take input and compute the output
    """

    def __init__(self):
        self.f = tanh
        self.f_prime = tanh_prime

    def forward(self, input):
        self.input = input
        return self.f(input)

    def backward(self, grad):
        return self.f_prime(self.input) * grad


class Tanh(Activation):
    def __init__(self):
        super().__init__()
