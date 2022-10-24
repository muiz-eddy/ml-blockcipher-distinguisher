from gc import callbacks
from tkinter import font
import tensorflow as tf
import numpy as np
import data_generation as dg
import time
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.utils import shuffle



def train_model(neuron_1 = 128, neuron_2 = 1024, output = 2, sample_size = 10000, sample_test = 10000,epochs = 25, diff = (0x0700000000000700, 0x7000000000007000, 0x0070000000000070, 0x0070000000000070)):
    st = time.time()
    model = Sequential()
    model.add(Dense(neuron_1, input_shape = (1,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(neuron_2))
    model.add(Activation('relu'))

    model.add(Dense(output))
    model.add(Activation('softmax'))
    
    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['acc']) 
    # X, Y, Z = dg.td_speck(data_sample = sample_size, diff_c= diff)
    Data = pd.read_excel('training_data/speck_5.xlsx')
    Data = np.array(Data)
    X = Data[:,[0]]
    Y = Data[:,-1]

    X, Y = shuffle(X, Y)
    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=42)
    y_valid = to_categorical( y_valid)
    y_train = to_categorical(y_train)
    #print(Y_train)
    #print(Y_train.shape)
    
    #X_val, Y_val = dg.td_present(data_sample = sample_test, diff_c= diff)
    #Y_val1 = to_categorical(Y_val)
    #model.summary()
    #print(Y_train1.shape)
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = 1024, validation_data = (x_valid, y_valid), shuffle = True)

    #representing in graph
    np.save("model_accuracy/128_1024_result_speck_5_acc.npy", history.history['acc'])
    np.save("model_accuracy/128_1024_result_speck_5_valacc.npy", history.history['val_acc'])
    
    scores = model.evaluate(x_train, y_train, verbose=0)

    from sklearn.metrics import confusion_matrix,classification_report
    y_prediction = model.predict(x_valid)
    y_prediction = np.argmax (y_prediction, axis = 1)
    y_test=np.argmax(y_valid, axis=1)
    
    classification_rep = classification_report(y_test, y_prediction, target_names= ['0','1'])
    print(classification_rep)
    result = confusion_matrix(y_test, y_prediction)
    print(result)

    accuracy = "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
    model.save("trained_model/128_1024_speck_5.h5")
    
    et = time.time()
    elapsed_time = et-st
    print("Model Saved,", accuracy , "Elapsed Execution Time: ", elapsed_time, 'seconds')
    print("Max training accuracy: ", np.max(history.history['acc']))
    print("Min training accuracy: ", np.min(history.history['acc']))
    return model
    
 
def make_model(neuron_1 = 128, neuron_2 = 1024, output = 2, lr = 0.0001):
    model = Sequential()
    model.add(Dense(neuron_1, input_shape = (1,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(neuron_2))
    model.add(Activation('relu'))

    model.add(Dense(output))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['acc']) 

    return model

def one_train_model(epochs = 25, diff = [0x0700000000000700, 0x7000000000007000]):
    #read data from excel
    # Data = pd.read_excel('training_data/one_speck_3_test.xlsx')
    # Data = np.array(Data)
    X, Y = dg.td_present(data_sample = 2**10 , diff_c= diff)

    # X = Data[:,[0]]
    # Y = Data[:,-1]
    # print(X.shape)

    X, Y = shuffle(X, Y)
    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, shuffle= True, stratify=Y, random_state=42)
    y_valid = to_categorical( y_valid)
    y_train = to_categorical(y_train)
    
    model = make_model()

    model.fit(x_train, y_train, epochs = epochs, batch_size = 1024, validation_data = (x_valid, y_valid), shuffle = True)

    return model

test = train_model()