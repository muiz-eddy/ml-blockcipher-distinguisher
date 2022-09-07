import os
import data_generation as dg
import numpy as np
import tensorflow
import pickle


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler


def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_model(neurons1 = 1024, neurons2 = 1024, blocks = 2, word_size = 16, layer = 3, output = 1):
    model = Sequential()
    model.add(Dense(neurons1, activation = "relu", input_shape = (blocks * word_size * 2, )))
    for i in range(layer):
        model.add(Dense(neurons2, activation = "relu"))
    
    model.add(Dense(output, activation = "sigmoid"))
    return(model)

def training(trained_model = "new", epochs_num = 25, neuron = 32, neuron2 = 1024, layer = 3, data_sample = 2**5, diff_c = [79042080,100000]):
    file_n = "speck_"
    if (trained_model == "new"):
        mod = make_model(layer = layer, neurons1 = neuron, blocks = 2, neurons2 = neuron2, word_size = int(16/2))
        mod.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    else:
        mod = load_model(trained_model)

        X_train, Y_train = dg.td_speck(data_sample = data_sample, diff_c= diff_c)
        X_val, Y_val = dg.td_speck(data_sample = data_sample, diff_c= diff_c)
        
    mod.summary()    
    
    if (trained_model == "new"):
        mod = save_model(mod,file_n + ".h5")
        print("saved in: ", file_n + ".h5")
    
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    model_fit = mod.fit(X_train,Y_train, epochs = epochs_num, batch_size = 100, validation_data = (X_val, Y_val), callbacks = [lr])
    np.save(file_n + 'model_fit', model_fit.history['val_acc'])
    file = open(file_n + 'model_fit' + 'history', 'wb')
    pickle.dump(model_fit.history, file)
    print("Best validation Accuracy: ", np.max(model_fit.history['val_acc']))
    return(mod, model_fit)

test = training(trained_model = "new", epochs_num = 25, neuron = 32, neuron2 = 1024, layer = 3, data_sample = 2**5, diff_c = [79042080,100000])

print(test)
    
        


