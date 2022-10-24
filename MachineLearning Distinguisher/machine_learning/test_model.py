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


loaded_model = load_model("trained_model/128_1024_speck_1.h5")

X,Y,Z = dg.td_speck(2**10, (0x79042080, 0x100000))

    

Y = to_categorical(Y)
train = loaded_model.evaluate(X, Y, verbose=0)

predict_x=loaded_model.predict(X) 
classes_x=np.argmax(predict_x,axis=1)


# for i in range(len(X)):
#     print("X=%s, Predicted=%s" % (X[i], classes_x[i]))

print("%s: %.2f%%" % (loaded_model.metrics_names[1], train[1]*100))