import numpy as np 

def init():
    np.random.seed(1)
    W = [2*np.random.random((4,3))-1, 2*np.random.random((1,4))-1]
    B = [np.zeros((4,1)), np.zeros((1,1))]
    return W, B

def forward_prop(X, W, B, activation):
    l1 = activation(np.dot(X,W[0].T))
    l2 = activation(np.dot(l1,W[1].T))
    return [l1, l2]

def back_prop(Y, A, W, B, derivation):
    l2_error = Y-A[1]
    l2_delta = l2_error*derivation(A[1])
    l1_error = l2_delta.dot(W[1])
    l1_delta = l1_error*derivation(A[0])
    W[1] += A[1].T.dot(l2_delta)
    W[0] += A[0].T.dot(l1_delta)    
    return np.mean(np.abs(l2_error)) 