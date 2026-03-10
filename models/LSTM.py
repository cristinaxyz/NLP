import numpy as np
import random
import math

def sigmoid(x):
    """
    Binary classification. Calculates the probability.
    
    :param x: Description
    """
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    """
    part of backpropagation
    
    :param values: Description
    """
    return values*(1-values)

def tanh_derivative(values):
    """
    part of backpropagation
    
    :param values: Description
    """
    return 1. - values ** 2

def rand_arr(a, b, *args): 
    """
    createst uniform random array w/ values in [a,b) and shape args
    
    :param a: int
    :param b: int
    :param args: Description
    """
    np.random.seed(0) 
    #rescales numbers from [0,1] to [a,b]
    return np.random.rand(*args) * (b - a) + a

def lstm_model(train_ds, dev_ds, test_ds, seed):
    """
    Docstring for lstm_model
    
    :param train_ds: Description
    :param dev_ds: Description
    :param test_ds: Description
    :param seed: Description
    """
    #input gate -- lets in optional information necessary from the current cell state.
    #output gate -- updates and finalizes the next hidden state
    #forget gate -- eliminates unnecessary information
    pass