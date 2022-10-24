from cgi import test
from imp import C_BUILTIN
from speck import SpeckCipher
from Present import encryption
import sys
import os
import numpy as np

parent_dir = os.getcwd()
sys.path.append(parent_dir)


# input differences taken from Machine Learning Attacks On SPECK baksi et.
def td_speck2(data_sample=10000, diff_c=[79042080, 100000]):
    X = np.empty((0,9), int)
    P = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext = P.tolist()
    Keys = K.tolist()
    P2 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K2 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext2 = P2.tolist()
    Keys2 = K2.tolist()
    P3 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K3 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext3 = P3.tolist()
    Keys3 = K3.tolist()
    P4 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K4 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext4 = P4.tolist()
    Keys4 = K4.tolist()
    P5 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K5 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext5 = P5.tolist()
    Keys5 = K5.tolist()
    P6 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K6 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext6 = P6.tolist()
    Keys6 = K6.tolist()
    P7 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K7 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext7 = P7.tolist()
    Keys7 = K7.tolist()
    P8 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K8 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext8 = P8.tolist()
    Keys8 = K8.tolist()
    for i in range(len(diff_c)):
        P1 = P ^ diff_c[i]
        P1 = P1.tolist()
        P_ = P2 ^ diff_c[i]
        P_ = P_.tolist()
        P_1 = P3 ^ diff_c[i]
        P_1= P_1.tolist()
        P_2 = P4 ^ diff_c[i]
        P_2= P_2.tolist()
        P_3 = P5 ^ diff_c[i]
        P_3= P_3.tolist()
        P_4 = P6 ^ diff_c[i]
        P_4= P_4.tolist()
        P_5 = P7 ^ diff_c[i]
        P_5= P_5.tolist()
        P_6 = P8 ^ diff_c[i]
        P_6= P_6.tolist()
        for j in range(data_sample):
            C = SpeckCipher(Keys[j], 64, 32, 'ECB').encrypt(Plaintext[j])
            C1 = SpeckCipher(Keys[j], 64, 32, 'ECB').encrypt(P1[j])
            C2 = SpeckCipher(Keys2[j], 64, 32, 'ECB').encrypt(Plaintext2[j])
            C_ = SpeckCipher(Keys2[j], 64, 32, 'ECB').encrypt(P_[j])
            C3 = SpeckCipher(Keys3[j], 64, 32, 'ECB').encrypt(Plaintext3[j])
            C_1 = SpeckCipher(Keys3[j], 64, 32, 'ECB').encrypt(P_1[j])
            C4 = SpeckCipher(Keys4[j], 64, 32, 'ECB').encrypt(Plaintext4[j])
            C_2 = SpeckCipher(Keys4[j], 64, 32, 'ECB').encrypt(P_2[j])
            C5 = SpeckCipher(Keys5[j], 64, 32, 'ECB').encrypt(Plaintext5[j])
            C_3 = SpeckCipher(Keys5[j], 64, 32, 'ECB').encrypt(P_3[j])
            C6 = SpeckCipher(Keys6[j], 64, 32, 'ECB').encrypt(Plaintext6[j])
            C_4 = SpeckCipher(Keys6[j], 64, 32, 'ECB').encrypt(P_4[j])
            C7 = SpeckCipher(Keys7[j], 64, 32, 'ECB').encrypt(Plaintext7[j])
            C_5 = SpeckCipher(Keys7[j], 64, 32, 'ECB').encrypt(P_5[j])
            C8 = SpeckCipher(Keys8[j], 64, 32, 'ECB').encrypt(Plaintext8[j])
            C_6 = SpeckCipher(Keys8[j], 64, 32, 'ECB').encrypt(P_6[j])
            X = np.append(X, np.array([[C ^ C1,C2 ^ C_, C3 ^ C_1, C4 ^ C_2, C5 ^ C_3, C6 ^ C_4, C7 ^ C_5, C8 ^ C_6, i]]), axis=0)
    X_train = X[:,[0,1,2,3,4,5,6,7]]
    Y_train = X[:,-1]
    return X_train, Y_train

def td_present2(data_sample=10000, diff_c=[0x0700000000000700, 0x7000000000007000, 0x0070000000000070, 0x0070000000000070]):
    X = np.empty((0,9), int)
    P = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext = P.tolist()
    Keys = K.tolist()
    P2 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K2 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext2 = P2.tolist()
    Keys2 = K2.tolist()
    P3 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K3 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext3 = P3.tolist()
    Keys3 = K3.tolist()
    P4 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K4 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext4 = P4.tolist()
    Keys4 = K4.tolist()
    P5 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K5 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext5 = P5.tolist()
    Keys5 = K5.tolist()
    P6 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K6 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext6 = P6.tolist()
    Keys6 = K6.tolist()
    P7 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K7 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext7 = P7.tolist()
    Keys7 = K7.tolist()
    P8 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K8 = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext8 = P8.tolist()
    Keys8 = K8.tolist()
    for j in range(len(diff_c)):
        P1 = P ^ diff_c[j]
        P1 = P1.tolist()
        P_ = P2 ^ diff_c[j]
        P_ = P_.tolist()
        P_1 = P3 ^ diff_c[j]
        P_1= P_1.tolist()
        P_2 = P4 ^ diff_c[j]
        P_2= P_2.tolist()
        P_3 = P5 ^ diff_c[j]
        P_3= P_3.tolist()
        P_4 = P6 ^ diff_c[j]
        P_4= P_4.tolist()
        P_5 = P7 ^ diff_c[j]
        P_5= P_5.tolist()
        P_6 = P8 ^ diff_c[j]
        P_6= P_6.tolist()
        for z in range(data_sample):
            C = encryption(Plaintext[z], Keys[z], 4)
            C1 = encryption(P1[z], Keys[z], 4)
            C2 = encryption(Plaintext2[z], Keys2[z], 4)
            C_ = encryption(P_[z], Keys2[z], 4)
            C3 = encryption(Plaintext3[z], Keys3[z], 4)
            C_1 = encryption(P_1[z], Keys3[z], 4)
            C4 = encryption(Plaintext4[z], Keys4[z], 4)
            C_2 = encryption(P_2[z], Keys4[z], 4)
            C5 = encryption(Plaintext5[z], Keys5[z], 4)
            C_3 = encryption(P_3[z], Keys5[z], 4)
            C6 = encryption(Plaintext6[z], Keys6[z], 4)
            C_4 = encryption(P_4[z], Keys6[z], 4)
            C7 = encryption(Plaintext7[z], Keys7[z], 4)
            C_5 = encryption(P_5[z], Keys7[z], 4)
            C8 = encryption(Plaintext8[z], Keys8[z], 4)
            C_6 = encryption(P_6[z], Keys8[z], 4)
            X = np.append(X, np.array([[C ^ C1,C2 ^ C_, C3 ^ C_1, C4 ^ C_2, C5 ^ C_3, C6 ^ C_4, C7 ^ C_5, C8 ^ C_6, j]]), axis=0)
            
    X_train = X[:,[0,1,2,3,4,5,6,7]]
    Y_train = X[:,-1]
    return X_train,Y_train
