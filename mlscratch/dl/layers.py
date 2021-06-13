import numpy as np


class Layers:
    """
    Base class for layers
    """

    def __init__(self):
        self.params = {}

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layers):
    def __init__(self, input_size, neurons):
        """
        Make a params dictionary and store weights and bias
        """
        super().__init__()
        self.input_size = input_size
        self.neurons = neurons
        self.params["w"] = np.random.randn(input_size, neurons)
        self.params["b"] = np.random.randn(1, neurons)
        self.grads = {}

    def forward(self, input_data):
        """
        Linear Output
        """
        self.input_data = input_data
        # print(input_data.shape,self.params["w"].shape,self.params["b"].shape)
        # print((input_data @ self.params["w"] + self.params["b"]).shape)
        # print(input_data @ self.params["w"] + self.params["b"])
        return np.matmul(input_data, self.params["w"]) + self.params["b"]

    def backward(self, grad):
        """
        Bacpropogation for linear layer
        inflow gradient * local gradient
        local gradient = input
        """
        self.grads["w"] = np.matmul(self.input_data.T, grad)
        self.grads["b"] = np.sum(grad, axis=0)
        return np.matmul(self.input_data.T, grad)
