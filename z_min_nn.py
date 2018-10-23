import numpy as np
import data_sets as ds
import print_funcs as pf
import model_11lines as m11
import activation_funcs as af

#m, X, Y = ds.xor()
m, X, Y = ds.one_col_correlation()

pf.arr_vec_print("X (examples x inputs) =", X, "Y =", Y)

W, B = m11.init()

pf.arr_vec_print("L0 (neurons x connections) =", W[0], "Bias =", B[0])
pf.arr_vec_print("L1 (neurons x connections) =", W[1], "Bias =", B[1])

def my_back_prop(m, Y, A, W, B, der_from_A):
    dZ1 = A[1]-Y
    assert len(dZ1)==5 # examples

    dW1 = np.dot(dZ1.T, A[0]) / m
    assert dW1.shape==(1,4) # connections to last neuron

    dB1 = np.sum(dZ1) / m
    assert isinstance(dB1, float)

    dZ0 = np.dot(dZ1,W[1]) * der_from_A(A[0])
    assert dZ0.shape==(5,4)

    dW0 = np.dot(dZ0.T,X) / m
    assert dW0.shape==(4,3)
    
    dB0 = np.sum(dZ0) / m    
    assert isinstance(dB0, float)

    W[1] -= dW1
    B[1] -= dB1
    W[0] -= dW0
    B[0] -= dB0

    return np.average(np.absolute(dZ1))

for i in range(0,10001):
    A = m11.forward_prop(X, W, B, af.sigmoid)
    error = my_back_prop(m, Y, A, W, B, af.der_from_simgoid)
    if(error<.01): break
    if(i%1000==0): print("Error after "+str(i)+" iterations: "+'{0:8.4f}'.format(error))

print("After "+str(i)+" iterations, the error is "+
    '{0:8.4f}'.format(error)+" and the prediction is:")
pf.arr_print("A[1].T =", A[1].T)
