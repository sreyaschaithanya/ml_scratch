from mlscratch.dl.activation import Tanh
from mlscratch.dl.layers import Dense
from mlscratch.dl.net import Net
import numpy as np

df = np.random.randn(5,2) * 10
print(df)

network = Net()

network.add(Dense(2,6))
network.add(Tanh())
network.add(Dense(6,4))
network.add(Tanh())

#network.config()

out = network.forward(df)

print(out)