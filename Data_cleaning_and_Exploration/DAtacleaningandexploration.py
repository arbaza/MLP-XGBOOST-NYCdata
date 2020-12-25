
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[37]:
mydir = os.getcwd()

os.chdir('..')
os.chdir('..')
mydir = os.getcwd()
mydir_tmp = mydir + "\Dataset\Input_Datasets"
mydir_new = os.chdir(mydir_tmp)


#reading the 2 dataset and concatenating them
df1 = pd.read_csv('green_tripdata_2018-02.csv')
df2 = pd.read_csv('green_tripdata_2018-03.csv')
frames = [df1, df2]
df = pd.concat(frames)
df.head()

os.chdir('..')
os.chdir('..')

# In[31]:


df.count()


# In[32]:


#dropping the parameter as it has no value
df = df.drop('ehail_fee', axis=1)


# In[33]:


df['store_and_fwd_flag'].value_counts()


# In[34]:


#converting the flag to 1 and 0
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].apply(lambda x: 1 if x == 'N' else 0)
df = df.rename(columns={'store_and_fwd_flag':'dispatch'})
df['dispatch'].value_counts()


# In[36]:



df.head(5)


# In[20]:


#extracting the date and time of the pickup and dropoff date time
df['PUD'] = pd.to_datetime(df['lpep_pickup_datetime']).dt.date
df['PUT'] = pd.to_datetime(df['lpep_pickup_datetime']).dt.time
df['DOD'] = pd.to_datetime(df['lpep_dropoff_datetime']).dt.date
df['DOT'] = pd.to_datetime(df['lpep_dropoff_datetime']).dt.time


# In[21]:


#Grouping long trips
long_trips = df[df['PUD'] != df['DOD']]
print("Count of long trips: ", long_trips.count())


# In[22]:


agg = df.groupby(df['PUD']).aggregate('sum')


# In[38]:


#Plotting the dispatches made each day
plt.figure(figsize=(18,5))
ax = plt.gca()
ax.set_xlabel('Days',fontsize=20)
ax.set_ylabel('Number of dispatches',fontsize=20)
ax.set_title('Number of dispatches for Feb 2018 and Mar 2018',fontsize=20, color='brown')

plt.plot(agg['dispatch'])


# In[39]:


#Plotting the number of passengers picked up at each location
location_pattern = df.groupby(df['PULocationID']).aggregate('sum')
plt.figure(figsize=(18,5))
ax = plt.gca()
ax.set_xlabel('Pick-up location ID',fontsize=20)
ax.set_xlim(0,df['PULocationID'].max()+ 5)
ax.set_ylabel('Number of passengers',fontsize=20)
print("location_pattern['passenger_count'].max(): ", location_pattern['passenger_count'].max())
ax.set_ylim(0,location_pattern['passenger_count'].max()+ 10000)
ax.set_title('Number of passengers picked up at a particular location for 2 months',fontsize=20, color='brown')
plt.plot(location_pattern['passenger_count'])


# In[40]:


#plotting the number of passengers in a trip
passenger_count = df.groupby(df['passenger_count']).size()
plt.figure(figsize=(10,5))
ax = plt.gca()
ax.set_xlabel('Number of passengers in a ride',fontsize=20)
ax.set_ylabel('Number of dispatches/trips',fontsize=20)
ax.set_title('Number of passengers in a trip vs total number of dispatches/trips made',fontsize=20, color='brown')
ax = passenger_count.plot.bar(color = 'grey')


# In[ ]:


#saving the clean data in csv format
df.to_csv('green_clean_2018.csv', index=False)
df.info()
