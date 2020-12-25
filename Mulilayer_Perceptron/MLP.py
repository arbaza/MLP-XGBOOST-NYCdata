
# coding: utf-8

# In[32]:

import os
from numpy import array
from numpy import argmax
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671)


# In[33]:


#defining the function that will be used to display the graph for loss and number of epochs
def graph_plot(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


# In[34]:

mydir = os.getcwd()
mydir = os.chdir('..')
mydir = os.chdir('..')
mydir = os.getcwd()
mydir_tmp = mydir + "\Dataset\Input_Datasets"
mydir_new = os.chdir(mydir_tmp)

#reading the dataset
dataset = pd.read_csv('green_clean_2018.csv')
dataset.info()

mydir = os.chdir('..')
mydir = os.chdir('..')
# In[35]:


#dropping unwanted parameters as these parameters have been cleaned and saved in new variables
X = dataset.drop(columns=['dispatch','trip_type', 'VendorID','improvement_surcharge','payment_type','mta_tax', 
                         'RatecodeID','tolls_amount','PUT','DOT','lpep_pickup_datetime', 'lpep_dropoff_datetime', 'PUD', 'DOD'],axis=1)
y = dataset['PULocationID']

X.head()


# In[36]:


#setting up the parameters
NB_EPOCH = 20
BATCH_SIZE = 128
VALIDATION_SPLIT=0.2
VERBOSE = 1
OPTIMIZER = SGD() 
N_HIDDEN = 128
DROPOUT = 0.3


# In[37]:


#Defining the one hot encoding function 
def OneHotEncoding(y):
    data = array(y)
    encoded = to_categorical(data)
    print(np.shape(encoded))
    return encoded


# In[38]:


#definning the architecture of the mlp model
def model(N_HIDDEN,RESHAPED,NB_CLASSES,OPTIMIZER,DROPOUT):
    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
    model.add(Activation('relu'))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    #model.add(Dropout(DROPOUT))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
    optimizer=OPTIMIZER,
    metrics=['accuracy'])
    return model


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
RESHAPED = len(X_train.columns)


# In[40]:


#getting y_train from the function one hot encoding
y_train = OneHotEncoding(y_train)
NB_CLASSES = y_train.shape[1]


# In[41]:


#showing the model
model= model(N_HIDDEN,RESHAPED,NB_CLASSES,OPTIMIZER,DROPOUT)


# In[42]:


#running the model
history = model.fit(X_train, y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)


# In[43]:


#plotting using the graph_plot function
graph_plot(history)


# In[44]:


y_test = OneHotEncoding(y_test)


# In[45]:


#printing the accuracy of the model
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

