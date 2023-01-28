import os
import data_generation as dg
import tensorflow
import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

print("1. TEST SPECK \n" + "2. TEST PRESENT \n")
s = input("Enter input: ")
if(int(s) == 1):
    userin = input("Filename: ")
    loaded_model = load_model("trained_model/" + userin + ".h5")
    print("1 - [0x79042080, 0x100000]")
    print("2 - [0x0700000000000700, 0x7000000000007000]")
    diff = input("Enter Input: ")
    if int(diff) == 1:
        X,Y,Z = dg.td_speck(data_sample = 2000, diff_c= [0x79042080, 0x100000])
        
    else:
        X,Y,Z = dg.td_speck(data_sample = 2**15, diff_c= [0x0700000000000700, 0x7000000000007000])
        

else:
    userin = input("Filename: ")
    loaded_model = load_model("trained_model/" + userin + ".h5")
    print("1 - [0x79042080, 0x100000]")
    print("2 - [0x0700000000000700, 0x7000000000007000]")
    diff = input("Enter Input: ")
    if int(diff) == 1:
        X,Y,Z = dg.td_present(data_sample = 2**15, diff_c= [0x79042080, 0x100000], rounds = 1)
        
    else:
        X,Y,Z = dg.td_present(data_sample = 2**15, diff_c= [0x0700000000000700, 0x7000000000007000], rounds = 1)
        

# X,Y,Z = dg.td_present(data_sample = 2**15,rounds = r,diff_c= [0x0700000000000700, 0x7000000000007000])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X) 
X = scaler.transform(X)

Y = to_categorical(Y)
train = loaded_model.evaluate(X, Y, verbose=0)

predict_x=loaded_model.predict(X) 

classes_x=np.argmax(predict_x,axis=1)
Y = np.argmax(Y, axis = 1)
# print(classes_x)

# for i in range(len(X)):
#     print("X=%s, Predicted=%s" % (X[i], classes_x[i]))
# print(accuracy_score(Y, classes_x))
testacc = train[1]*100
print("%s: %.2f%%" % (loaded_model.metrics_names[1], testacc))

trainacc = np.load('model_accuracy/' + userin + '_acc.npy')
trainacc = trainacc[-1]*100

if (testacc-5 <= trainacc <= testacc+5):
    print("\n Encryption is Cipher")

elif (testacc-3 <= 50 <= testacc+0.9):
    print("\n Encryption is Random")

else:
    print("\n Something is wrong with the training model")

