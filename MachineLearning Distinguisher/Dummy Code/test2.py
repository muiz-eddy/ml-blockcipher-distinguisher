from sklearn import datasets
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

iris = datasets.load_iris() #Loading the dataset
print(iris.keys())

print(iris['feature_names'])
print(iris['target'])
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )


from sklearn.model_selection import train_test_split

# Droping the target and species since we only need the measurements
X = iris.drop(['target'], axis=1)

print(X)
# converting into numpy array and assigning petal length and petal width
X = X.to_numpy()

y = iris['target']

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X,y)

training_prediction = log_reg.predict(X_train)
test_prediction = log_reg.predict(X_test)

from sklearn import metrics

print("Precision, Recall, Confusion matrix, in training\n")

# Precision Recall scores
print(metrics.classification_report(y_train, training_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_train, training_prediction))


print("Precision, Recall, Confusion matrix, in testing\n")

# Precision Recall scores
print(metrics.classification_report(y_test, test_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_test, test_prediction))


def train_model(neuron_1 = 128, neuron_2 = 1024, output = 3, sample_size = 2500, sample_test = 10000,epochs = 25, diff = (0x0700000000000700, 0x7000000000007000, 0x0070000000000070, 0x0070000000000070)):
    
    iris = datasets.load_iris() #Loading the dataset
    iris.keys()
    
    iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )
    # Droping the target and species since we only need the measurements
    X = iris.drop(['target'], axis=1)

    # converting into numpy array and assigning petal length and petal width
    X = X.to_numpy()[:, (2,3)]
    y = iris['target']
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)
    
    Y_train1 = to_categorical(y_train)
    Y_test1 = to_categorical(y_test)
    model = Sequential([

           
        #dense layer 1
        Dense(neuron_1, activation = 'relu', input_shape= (2,)),
        
        tf.keras.layers.BatchNormalization(),
        
        #dense layer 3
        Dense(neuron_2, activation = 'relu'),
        
        
        #Output layer
        Dense(output,activation = 'softmax'),
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['acc']) 
   
  
    model.fit(X_train, Y_train1, epochs = epochs, batch_size = 100, validation_data = [X_test, Y_test1])

    test = model.predict(X_test)
    classes_x=np.argmax(test,axis=1)

    for i in range(len(X_test)):
	    print("X=%s, Predicted=%s" % (X[i], classes_x[i]))

test = train_model()

