import numpy as np

def one_col_correlation():
    return (5,
        np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,1,1] ]),
        np.array([[0,0,1,1,1]]).T)

def xor():
    return (5,
        np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,1,1] ]), 
        np.array([[0,1,1,0,0]]).T)