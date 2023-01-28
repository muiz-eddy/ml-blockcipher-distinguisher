from gc import callbacks
from tkinter import font
import tensorflow as tf
import numpy as np
import data_generation as dg
from time import perf_counter
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot as plt
from sklearn.utils import shuffle

def make_model(neuron_1 = 128, neuron_2 = 1024, output = 2, lr = 0.0001):
    model = Sequential()
    model.add(Dense(neuron_1, input_shape = (1,)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(neuron_2))
    model.add(Activation('relu'))

    model.add(Dense(output))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy']) 

    return model

def train_model(neuron_1 = 128, neuron_2 = 1024, output = 2, sample_size = 10000, sample_test = 10000,epochs = 25, diff = (0x0700000000000700, 0x7000000000007000, 0x0070000000000070, 0x0070000000000070)):
    st = perf_counter()
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    model = make_model()
    # X, Y, Z = dg.td_speck(data_sample = sample_size, diff_c= diff)
    # Data = pd.read_excel('training_data/speck_3.xlsx')
    
    print("1 = Train SPECK dataset")
    print("2 = Train PRESENT dataset")
    
    userin = input("Enter Input: ")
    if (userin == str(userin)):
        print("1. SPECK Differential \n" + "2. PRESENT Differential \n")
        inp = input("Enter Input: ")
        if (int(inp) == 1):
            read = input("read Filename: ")
            Data = pd.read_csv('training_data/speck_speckdiff/' + read + '.csv')
            Data = np.array(Data)
        else:
            read = input("read Filename: ")
            Data = pd.read_csv('training_data/speck_presentdiff/' + read + '.csv')
            Data = np.array(Data)
    else:
        print("1. SPECK Differential \n" + "2. PRESENT Differential \n")
        inp = input("Enter Input: ")
        if (int(inp) == 1):
            read = input("read Filename: ")
            Data = pd.read_csv('training_data/present_speckdiff/' + read + '.csv')
            Data = np.array(Data)
        else:
            read = input("read Filename: ")
            Data = pd.read_csv('training_data/present_presentdiff/' + read + '.csv')
            Data = np.array(Data)
            
    X = Data[:,[0]]
    Y = Data[:,-1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X) 
    X = scaler.transform(X)
    X, Y = shuffle(X, Y)
    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=42)
    y_train = to_categorical( y_train)
    y_valid = to_categorical(y_valid)

    # Y = to_categorical(Y)
    #print(Y_train)
    #print(Y_train.shape)
    # x_valid, y_valid, Z = dg.td_present(data_sample = 2**15, rounds = r,diff_c= [0x0700000000000700, 0x7000000000007000])
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = scaler.fit(x_valid) 
    # x_valid = scaler.transform(x_valid)
    #model.summary()
    #print(Y_train1.shape)

    history = model.fit(x_train, y_train, epochs = epochs, batch_size = 1024, validation_data = (x_valid, y_valid), shuffle = True)

    #representing in graph
    np.save("model_accuracy/" + read + "_acc.npy", history.history['accuracy'])
    np.save("model_accuracy/" + read + "_valacc.npy", history.history['val_accuracy'])
    
    #scores = model.evaluate(x_train, y_train, verbose=0)
    valscores = model.evaluate(x_valid, y_valid, verbose = 0)

    from sklearn.metrics import confusion_matrix,classification_report
    y_prediction = model.predict(x_valid)
    y_prediction = np.argmax (y_prediction, axis = 1)
    y_test=np.argmax(y_valid, axis=1)
    
    classification_rep = classification_report(y_test, y_prediction, target_names= ['0','1'])
    print(classification_rep)
    result = confusion_matrix(y_test, y_prediction)
    print(result)

    #accuracy = "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
    valaccuracy = "%s: %.2f%%" % (model.metrics_names[1], valscores[1]*100)
    model.save("trained_model/" + read + ".h5")

    et = perf_counter()
    elapsed_time = et-st
    print("Model Saved,", "Elapsed Execution Time: ", elapsed_time, 'seconds')
    print("Final Epoch training accuracy: ", history.history['accuracy'][-1])
    print("Min training accuracy: ", np.min(history.history['accuracy']))
    print()
    print("Final Epoch Validation accuracy: ", history.history['val_accuracy'][-1])
    print("Min validation accuracy: ", np.min(history.history['val_accuracy']))
    return model

test = train_model()