from sklearn import datasets
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import data_generation as dg 


np.set_printoptions(threshold=sys.maxsize)

from sklearn.model_selection import train_test_split
diff = (0x0700000000000700, 0x7000000000007000, 0x0070000000000070, 0x0070000000000070)
test = dg.td_present2(data_sample = 200, diff_c= diff)

test = pd.DataFrame(
  data = np.c_[test[0], test[1]],
  
)


X = test.drop(test.columns[8], axis = 'columns')

X = X.to_numpy()


y = test[8]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(multi_class= 'multinomial')
log_reg.fit(X_train,y_train)


training_prediction = log_reg.predict(X_train)
test_prediction = log_reg.predict(X_test)
print(test_prediction)

from sklearn import metrics

print("Precision, Recall, Confusion matrix, in training\n")

# Precision Recall scores
print(metrics.classification_report(y_train, training_prediction, digits=4))

# # Confusion matrix
print(metrics.confusion_matrix(y_train, training_prediction))

print("Precision, Recall, Confusion matrix, in testing\n")

# Precision Recall scores
print(metrics.classification_report(y_test, test_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_test, test_prediction))