from speck import SpeckCipher
import sys
import os
import numpy as np

parent_dir = os.getcwd()
sys.path.append(parent_dir)


# input differences taken from Machine Learning Attacks On SPECK baksi et.
def td_speck(data_sample=2**2, diff_c=[79042080, 100000, 52030701, 8710609]):
    X = np.empty((0,2), int)
    P = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext = P.tolist()
    Keys = K.tolist()
    for i in range(len(diff_c)):
        P1 = P ^ diff_c[i]
        P1 = P1.tolist()
        for j in range(data_sample):
            C = SpeckCipher(Keys[j], 64, 32, 'ECB').encrypt(Plaintext[i])
            C1 = SpeckCipher(Keys[j], 64, 32, 'ECB').encrypt(P1[j])
            X = np.append(X, np.array([[C ^ C1,i]]), axis=0)
    X_train = X[:,0:1]
    Y_train = X[:,1]
    return X_train, Y_train
    

def test_data_speck(data_sample=2**2, diff_c=[79042080, 100000, 52030701, 8710609]):
    X = np.empty((0,2), int)
    P = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext = P.tolist()
    Keys = K.tolist()
    for i in range(len(diff_c)):
        P1 = P ^ diff_c[i]
        P1 = P1.tolist()
        


