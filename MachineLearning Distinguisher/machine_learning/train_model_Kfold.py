#THIS IS EXTRA FILE FOR VALIDATION PURPOSES, THE MAIN TRAINING FILE IS train_model.py

from gc import callbacks
from tkinter import font
import tensorflow as tf
import numpy as np
import data_generation as dg
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape = (1,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    

    model.add(Dense(1024))
    model.add(Activation('relu'))
    
    

    model.add(Dense(2))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)

    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['acc'])

    return model
def train_model(neuron_1 = 128, neuron_2 = 1024, output = 2, sample_size = 10000, sample_test = 10000,epochs = 25, diff = (0x0700000000000700, 0x7000000000007000, 0x0070000000000070, 0x0070000000000070)):
    st = time.time()
    
    
    X, Y, Z = dg.td_speck(data_sample = sample_size, diff_c= diff)
    
    
    for kfold, (train, test) in enumerate(KFold(n_splits=3, shuffle=True).split(X)):
        print(test)
        X_train,X_test = X[train],X[test]
        Y_train, Y_test = Y[train], Y[test]
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)
        model = create_model()
        model.fit(X_train, Y_train, epochs = epochs, batch_size = 1024, validation_data = (X_test, Y_test),shuffle = True)
        et = time.time()
        elapsed_time = et-st
        print('Model evaluation ',model.evaluate(X_test,Y_test))
        print("Model Saved,", "Elapsed Execution Time: ", elapsed_time, 'seconds')
        model.save_weights("speck_6.h5")
    
    
    # x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, shuffle= True, stratify=Y, random_state=42)
    # y_valid = to_categorical( y_valid)
    # y_train = to_categorical(y_train)
    #print(Y_train)
    #print(Y_train.shape)
    
    #X_val, Y_val = dg.td_present(data_sample = sample_test, diff_c= diff)
    #Y_val1 = to_categorical(Y_val)
    #model.summary()
    #print(Y_train1.shape)
        
        # scores = model.evaluate(x_train, y_train, verbose=0)

        # from sklearn.metrics import confusion_matrix,classification_report
        # y_prediction = model.predict(x_valid)
        # y_prediction = np.argmax (y_prediction, axis = 1)
        # y_test=np.argmax(y_valid, axis=1)
        
        # classification_rep = classification_report(y_test, y_prediction, target_names= ['0','1'])
        # print(classification_rep)
        # result = confusion_matrix(y_test, y_prediction)
        # print(result)
    # for i in range(len(x_valid)):
    #     print("X=%s, Predicted=%s, Actual=%s" % (x_valid[i], y_prediction[i], y_valid[i]))

    # import seaborn as sn
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import plot_confusion_matrix
    # df_cm = pd.DataFrame(result, range(2), range(2))
    # sn.set(font_scale = 1.4)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    # plt.show()

    # accuracy = "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
    
    
    return model
    
 
out = train_model(sample_size = 2**15, diff = [0x79042080, 0x100000])

print(out)