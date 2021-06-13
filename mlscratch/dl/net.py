import numpy as np
from itertools import islice

import mlscratch.dl.layers
import mlscratch.dl.loss


class Net:
    def __init__(self):
        self.net_layers = []
        self.layerOut = []

    def add(self, layer):
        self.net_layers.append(layer)

    def config(self):
        for lay in self.net_layers:
            print(lay.__dict__)

    def forward(self, input):
        enum_layer = enumerate(self.net_layers)
        for i, layer in enum_layer:
            if i == 0:
                layer_out = layer.forward(input)
                self.layerOut.append(layer_out)
                continue
            layer_out = layer.forward(self.layerOut[i - 1])
            self.layerOut.append(layer_out)
        return self.layerOut[-1]

    def backward(self, grad):
        pass
