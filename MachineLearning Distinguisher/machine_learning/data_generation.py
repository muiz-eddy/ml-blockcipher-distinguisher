import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.getcwd()
sys.path.append(parent_dir)
from speck import SpeckCipher
from Present import encryption
from tqdm import tqdm

def randomgen (data_sample = 10, bytes = 8):
    X = np.empty((0), object)
    for i in tqdm(range(data_sample)):
        a = int.from_bytes(os.urandom(bytes), sys.byteorder, signed = False)
        X = np.append(X, np.array([a]), axis=0)

    return X

################################################### MODEL 1 Multiple Input Difference ###############################################
def td_speck(data_sample=2, diff_c=[0x79042080, 0x100000]):
    X = np.empty((0,2), float)

    #Generate random arrays of Plaintext and Keys
    P = randomgen(data_sample, bytes = 4)
    K = randomgen(data_sample, bytes = 8)
    
    for j in range(len(diff_c)): #Loop based on the size of input difference
        P1 = P ^ diff_c[j]

        for z in range(data_sample): #loop based on data sample size
            C = SpeckCipher(K[z], 64, 32, 'ECB').encrypt(P[z])
            C1 = SpeckCipher(K[z], 64, 32, 'ECB').encrypt(P1[z])
            X = np.append(X, np.array([[C ^ C1, j]]), axis=0)
           
    X_train = X[:,[0]]
    Y_train = X[:,-1]
    return X_train,Y_train,X

def td_present(data_sample=2, diff_c=[0x0700000000000700, 0x7000000000007000,0x0070000000000070,0x0007000000000007], rounds = 1):
    X = np.empty((0,2), float)
    P = randomgen(data_sample, bytes = 8)
    K = randomgen(data_sample, bytes = 10)

    for j in range(len(diff_c)):
        P1 = P ^ diff_c[j]
        P1 = P1.tolist()
        for z in tqdm(range(data_sample)):
            C = encryption(P[z], K[z], rounds)
            C1 = encryption(P1[z], K[z], rounds)
            X = np.append(X, np.array([[C ^ C1, j]]), axis=0)
            
    X_train = X[:,[0]]
    Y_train = X[:,-1]
    return X_train,Y_train, X

def test_data_present(data_sample=2**2, diff_c=[0x0700000000000700, 0x7000000000007000]):
    X = np.empty((0,2), float)
    P = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)
    Plaintext = P.tolist()
    Keys = K.tolist()
    for j in range(len(diff_c)):
        P1 = P ^ diff_c[j]
        P1 = P1.tolist()
        for z in range(data_sample):
            C = encryption(Plaintext[z], Keys[z], 4)
            C1 = encryption(P1[z], Keys[z], 4)
            X = np.append(X, np.array([[C ^ C1, j]]), axis=0)
    X_test = X[:,[0]]
    Y_test = X[:,-1]
    return X_test, Y_test


# x,y, z= td_present(50000)

# df = pd.DataFrame(z)

# filepath = 'training_data/speck_test.xlsx'

# df.to_excel(filepath, index = False)


################################################### MODEL 2 One Input Difference ################################################
def one_td_speck(data_sample = 2**10, diff = 0x100000):
    X = np.empty((0,2), int) 
    P = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    P1 = np.frombuffer(os.urandom(4*data_sample), dtype=np.uint32)
    K = np.frombuffer(os.urandom(8*data_sample), dtype=np.uint64)

    P2 = P1 ^ diff
    PT = P.tolist()
    PT1 = P1.tolist()
    PT2 = P2.tolist()
    Keys = K.tolist()
    for i in range(data_sample):
        C0 = SpeckCipher(Keys[i], 64, 32, 'ECB').encrypt(PT[i])
        C1 = SpeckCipher(Keys[i], 64, 32, 'ECB').encrypt(PT1[i])
        C2 = SpeckCipher(Keys[i], 64, 32, 'ECB').encrypt(PT2[i])
        X = np.append(X, np.array([[C0 or C1, 0]]), axis=0)
        X = np.append(X, np.array([[C0 or C2, 1]]), axis=0)

    X_test = X[:,[0]]
    Y_test = X[:,-1]

    return X_test, Y_test, X, 

def main ():

    print("1 = Generate SPECK dataset")
    print("2 = Generate PRESENT dataset")
    
    userin = input("Select action: ")
    print()
    if userin == str(1):
        data = input("Data Size: ")
        print("differentials: ")
        print("1 - [0x79042080, 0x100000]")
        print("2 - [0x0700000000000700, 0x7000000000007000]")
        diff = input("Enter Input: ")
        if int(diff) == 1:
            X,Y,Z = td_speck(int(data), diff_c= [0x79042080, 0x100000])
            df = pd.DataFrame(Z)
            fl = input("filename: ")
            filepath = 'training_data/speck_speckdiff/' + fl + '.csv'
            df.to_csv(filepath, index = False)
        else:
            X,Y,Z = td_speck(int(data), diff_c=[0x0700000000000700, 0x7000000000007000])
            df = pd.DataFrame(Z)
            fl = input("filename: ")
            filepath = 'training_data/speck_presentdiff/' + fl + '.csv'
            df.to_csv(filepath, index = False)

    elif userin == str(2):
        data = input("Data Size: ")
        print("differentials: ")
        print("1 - [0x79042080, 0x100000]")
        print("2 - [0x0700000000000700, 0x7000000000007000]")
        diff = input("Enter Input: ")
        if int(diff) == 1:
            X,Y,Z = td_present(int(data), diff_c= [0x79042080, 0x100000], rounds = 1)
            df = pd.DataFrame(Z)
            fl = input("filename: ")
            filepath = 'training_data/present_speckdiff/' + fl + '.csv'
            df.to_csv(filepath, index = False)
        else:
            X,Y,Z = td_present(int(data), diff_c=[0x0700000000000700, 0x7000000000007000], rounds = 1)
            df = pd.DataFrame(Z)
            fl = input("filename: ")
            filepath = 'training_data/present_presentdiff/' + fl + '.csv'
            df.to_csv(filepath, index = False)

if __name__ == "__main__":
    main()

# x,y,z = td_speck(2**2)

# df = pd.DataFrame(z)

# filepath = 'training_data/speck_speckdiff/presentation.csv'

# df.to_csv(filepath, index = False)
