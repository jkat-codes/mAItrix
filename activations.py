import numpy as np

def sigmoid(Z): 
    return 1 / (1 + np.exp(-Z))

def softmax(Z): 
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_vals = np.exp(Z_shifted)
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

def relu(Z): 
    return np.maximum(0, Z)

def tanh(x): 
    return np.tanh(x)

def leaky_relu(Z, cache): 
    return np.maximum(0.01 * Z, Z)

## 0.01 in the function above and below can be changed

def deriv_leaky_relu(Z, cache): 
    return np.where(Z > 0, 1, 0.01)

def deriv_sigmoid(dA, cache): 
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ

def deriv_relu(Z): 
    return Z > 0


def deriv_tanh(x): 
    return (1 - np.power(x, 2))

def one_hot(Y): 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y