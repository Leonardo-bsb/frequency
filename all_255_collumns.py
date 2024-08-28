import numpy as np

def all_255(array):
    return np.argwhere(np.min(array,axis=0)==255)
