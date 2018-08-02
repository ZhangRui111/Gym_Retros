import numpy as np


def binary_array_to_int(array, size):
    out_list = []
    for i in range(size):
        input = array[i]
        out = 0
        for bit in input:
            out = (out << 1) | bit
        out_list.append(out)
    out_arr = np.array(out_list).reshape((size, 1))
    return out_arr


def simple_binary_array_to_int(array):
    input = array
    out = 0
    for bit in input:
        out = (out << 1) | bit
    return out
