import pandas as pd
import numpy as np
import os
import sys

def randomgen (data_sample = 10, bytes = 8):
    X = np.empty((0), object)
    for i in range(data_sample):
        a = int.from_bytes(os.urandom(bytes), sys.byteorder, signed = False)
        X = np.append(X, np.array([a]), axis=0)

    return X


