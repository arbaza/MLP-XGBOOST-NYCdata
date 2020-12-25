
# coding: utf-8

# In[1]:

import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[6]:
mydir = os.getcwd()
mydir = os.chdir('..')
mydir = os.chdir('..')
mydir = os.getcwd()
mydir_tmp = mydir + "\Dataset\Input_Datasets"
mydir_new = os.chdir(mydir_tmp)

#dataset read and dropping unwanted parameters as these parameters have been cleaned and saved in new variables
dataset = pd.read_csv('green_clean_2018.csv')
X = dataset.drop(columns=['PUT','DOT','lpep_pickup_datetime', 'lpep_dropoff_datetime', 'PUD', 'DOD', 'PULocationID'],axis=1)
y = dataset['PULocationID']
data_dmatrix = xgb.DMatrix(data=X,label=y)
mydir = os.chdir('..')
mydir = os.chdir('..')

# In[7]:


#using train_test_split from sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[8]:


X_train.info()


# In[9]:


#using xgbregressor from xgboost
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 1, learning_rate = 0.01,
                max_depth = 4, alpha = 10, n_estimators = 1)


# In[10]:


xg_reg.fit(X_train,y_train)


# In[11]:


preds = xg_reg.predict(X_test)


# In[12]:


#Calculating the rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# In[13]:


#setting the parameters
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.01,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[14]:


#printing the last rmse result
print((cv_results["test-rmse-mean"]).tail(1))


# In[15]:


print(cv_results)


# In[16]:


xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)


# In[17]:


#Plotting the feature importance
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


# In[18]:


X.info()


# In[21]:


#Using different features and using similar approach as above with different parameters and finding their respective parameters
X = dataset.drop(columns=['dispatch', 'day','trip_type', 'VendorID','improvement_surcharge','payment_type','mta_tax',
                         'RatecodeID','tolls_amount','month', 'lpep_pickup_datetime', 'lpep_dropoff_datetime', 'PUD', 'DOD','PUT','DOT', 'PULocationID'],axis=1)
y = dataset['PULocationID']
data_dmatrix = xgb.DMatrix(data=X,label=y)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[23]:


#Same parameters are used
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[24]:


#RMSE is calculated
#printscikit rfe((cv_results["test-rmse-mean"]).tail(1))


# In[26]:


#different parameters are dropped again
X = dataset.drop(columns=['VendorID','RatecodeID','mta_tax','tolls_amount','improvement_surcharge','total_amount','payment_type','trip_type','dispatch','PickUp_hr', 'lpep_pickup_datetime', 'lpep_dropoff_datetime', 'PUD', 'DOD','PUT','DOT', 'PULocationID'],axis=1)


# In[27]:


data_dmatrix = xgb.DMatrix(data=X,label=y)


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[29]:


#same parameters
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.01,
                'max_depth': 5, 'alpha': 10}


# In[30]:


cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[31]:


#RMSE value
print((cv_results["test-rmse-mean"]).tail(1))


# In[32]:


print(cv_results)

