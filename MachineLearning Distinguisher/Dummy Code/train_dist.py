import os
import data_generation as dg
import numpy as np
import tensorflow
import pickle


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import LearningRateScheduler

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_model(neurons1 = 1024, neurons2 = 1024, blocks = 2, output = 2):
    model = Sequential()
    model.add(Input(shape = (1,)))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(1024, activation = "relu"))
    model.add(Dense(1024, activation = "relu"))
    model.add(Dense(2, activation = "softmax"))
    return(model)

def training(trained_model = "new", epochs_num = 5, neuron = 32, neuron2 = 1024, layer = 3, data_sample = 2**5, diff_c = [79042080, 100000, 52030701, 8710609]):
    file_n = "speck_"
    if (trained_model == "new"):
        mod = make_model(neurons1 = neuron,neurons2 = neuron2)
        mod.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    else:
        mod = load_model(trained_model)

    X_train, Y_train = dg.td_speck(data_sample = data_sample, diff_c= diff_c)
    X_val, Y_val = dg.td_speck(data_sample = data_sample, diff_c= diff_c)
    print(X_train)
    print(Y_train)
    mod.summary()    
    
    #if (trained_model == "new"):
       # mod = save_model(mod,file_n + ".h5")
       # print("saved in: ", file_n + ".h5")
    
    lr = LearningRateScheduler(cyclic_lr(10, 0.02, 0.001))
    model_fit = mod.fit(X_train,Y_train, epochs = epochs_num, batch_size = 100, validation_data = (X_val, Y_val), callbacks = [lr], shuffle = True)
    #np.save(file_n + 'model_fit', model_fit.history['val_accuracy'])
    #file = open(file_n + 'model_fit' + 'history', 'wb')
    #pickle.dump(model_fit.history, file)
    #print("Best validation Accuracy: ", np.max(model_fit.history['val_accuracy']))
    return(mod, model_fit)

test = training(trained_model = "new", epochs_num = 20, neuron = 32, neuron2 = 1024, data_sample = 2**15, diff_c = [79042080, 100000])


print(test)
print("finish") 


#test on smaller model