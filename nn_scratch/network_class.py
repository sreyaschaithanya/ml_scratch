import numpy as np


class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[1:],sizes[:-1])]
    
    def FeedForward(self,a):
        for b,w in zip(self.bias,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def StochasticGradientDescent(self,train_data,epoch,batch_size,alpha):
        pass


def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def SigmoidGrad(x):
    return np.multiply(Sigmoid(x),(1-Sigmoid(x)))

def Tanh(x):
    return np.tanh(x)

def TanhGrad(x):
    return 1-np.multiply(Tanh(x),Tanh(x))

def RELU(x):
    return max(0,x)

def RELUGrad(x):
    if x>0:
        return 1
    else:
        return 0

def Leaky(x):
    return max(0.01*x,x)

def LeakyGrad(x):
    if x>0:
        return 1
    else:
        return 0.01