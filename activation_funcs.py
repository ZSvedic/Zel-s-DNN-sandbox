import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def der_from_simgoid(sig):
    return sig*(1-sig)