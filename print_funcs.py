import numpy as np

def _head_print(lhead, n, rhead):
    print(lhead)
    print("        ", end="")
    print(*[(str(i)).rjust(8) for i in range(0,n)], end="")
    print("  " + rhead.rjust(8))

def _fl_tab_print(index, list, myend="\n"):
    print(str(index).rjust(6)+": ", end="")
    print(*['{0:8.4f}'.format(i) for i in list], end=myend)

def arr_print(arr_text, arr):
    _head_print(arr_text, arr.shape[0], "")
    for i in range(0,arr.shape[1]):
        _fl_tab_print(i, arr[:,i])
    print()

def arr_vec_print(arr_text, arr, vect_txt, vec):
    assert arr.shape[0]==len(vec), "Array and vector need to have the same number of rows."
    _head_print(arr_text, arr.shape[1], vect_txt)
    for i in range(0,arr.shape[0]):
        _fl_tab_print(i, arr[i,:], "")
        print("  " + '{0:8.4f}'.format(vec[i,0]))
    print()
