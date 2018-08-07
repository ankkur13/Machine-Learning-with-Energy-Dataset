
# coding: utf-8

# # Energy Dataset Final Pipeline

# In[20]:


import pandas as pd
import time
import numpy as np
import datetime
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# In[21]:

print('LOG File name : pipeLine_logging.txt')
logfilename = 'pipeLine_logging.txt'
logging.basicConfig(filename=logfilename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('Program Started')
print('Program Started')


# In[22]:


logging.debug('Loading Data into Dataframe')
print('Loading Data into Dataframe')
try :  
    df_loaded = pd.read_csv("../home/energydata_complete.csv")
    df = df_loaded
    logging.debug('Data Size'+str(df.shape))
    print('Data Size'+str(df.shape))
    
except :
    logging.ERROR('Data logging failed')
    print('Data logging failed')


# In[23]:


logging.debug("Tranforming date time")
print("Tranforming date time")
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', utc=True)


# In[24]:


logging.debug('Creating Column NSM, week_status, day_of_week')
print('Creating Column NSM, week_status, day_of_week')
df['NSM'] = df.date.apply(lambda x: x.hour*3600 + x.minute*60 +x.second)
df['day_of_week'] = df.date.apply(lambda x: x.dayofweek)
df['week_status'] = df.day_of_week.apply(lambda x: 0 if (x == 5 or x == 6) else 1)


# In[25]:


shape_bool = df.date.nunique() == df.shape[0]
logging.debug('Checking if the date column is unique for each and every row to be: ')
logging.debug(shape_bool)
print('Checking if the date column is unique for each and every row to be: ', shape_bool)


# In[26]:


all_columns = df.columns.tolist()

logging.debug('Detecting Outliers for Each variable')
print('Detecting Outliers for Each variable')
df_describe = df.describe().T

logging.debug('Calculating Interquartile Range, Major Outlier and Minor Outlier')
print('Calculating Interquartile Range, Major Outlier and Minor Outlier')
df_describe['Interquartile Range'] = 1.5*(df_describe['75%'] - df_describe['25%'])
df_describe['Major Outlier'] = (df_describe['75%'] + df_describe['Interquartile Range'])
df_describe['Minor Outlier'] = (df_describe['25%'] - df_describe['Interquartile Range'])

logging.debug('Creating function to remove outliers')
print('Creating function to remove outliers')
def remove_outlier(df, variable):
    major_o = df_describe.loc[variable,'Major Outlier']
    minor_o = df_describe.loc[variable,'Minor Outlier']
    df = df.drop(df[(df[variable]>major_o) | (df[variable]<minor_o)].index)
    return df

outlier_column_list = [x for x in all_columns 
                       if x not in ('date', 'Appliances', 'lights')]

logging.debug('Removing Outliers')
print('Removing Outliers')
for column_name in outlier_column_list:
    df = remove_outlier(df, column_name)


# In[27]:


dropped = ((df_loaded.shape[0] - df.shape[0])/df_loaded.shape[0])*100
logging.debug('Percentage of Data Dropped: ')
logging.debug(dropped)
print('Percentage of Data Dropped: ', dropped)


# In[28]:


logging.debug('Transformation of WeekStatus and Days_of_week columns')
print('Transformation of WeekStatus and Days_of_week columns')
week_status = pd.get_dummies(df['week_status'], prefix = 'week_status')
day_of_week = pd.get_dummies(df['day_of_week'], prefix = 'day_of_week')

logging.debug('Concat dummy variable dataframe to the main dataframe')
print('Concat dummy variable dataframe to the main dataframe')
df = pd.concat((df,week_status),axis=1)
df = pd.concat((df,day_of_week),axis=1)

logging.debug('Droppin the WeekStatus and Day_of_week column')
print('Droppin the WeekStatus and Day_of_week column')
df = df.drop(['week_status','day_of_week'],axis=1)


# In[29]:


logging.debug('Renaming the column of dummy variables')
print('Renaming the column of dummy variables')
df = df.rename(columns={'week_status_0': 'Weekend', 'week_status_1': 'Weekday',
                   'day_of_week_0': 'Monday', 'day_of_week_1': 'Tuesday', 'day_of_week_2': 'Wednesday',
                  'day_of_week_3': 'Thursday', 'day_of_week_4': 'Friday', 'day_of_week_5': 'Saturday',
                  'day_of_week_6': 'Sunday'})


# In[30]:


logging.debug('Redefining the Appliances column, adding the consumption of lights and dropping it')
print('Redefining the Appliances column, adding the consumption of lights and dropping it')
df['Appliances'] = df['Appliances'] + df['lights']
df = df.drop(['lights'],axis=1)
df = df.drop(['date'],axis=1)


# In[ ]:


logging.debug('Loading libraries for feature selection and prediction')
print('Loading libraries for feature selection and prediction')
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


X = df.drop(['Appliances'],axis=1)
y = df['Appliances']

logging.debug('Splitting for Feature Selection')
print('Splitting for Feature Selection')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model_rf = RandomForestRegressor(n_estimators=10)

logging.debug('Feature selection as part of a pipeline')
print('Feature selection as part of a pipeline')
clf = Pipeline([('feature_selection', RFE(model_rf, n_features_to_select = 5)),
                ('classification', RandomForestRegressor())])

logging.debug('Fitting the random forest')
print('Fitting the random forest')
clf.fit(X_train, y_train)

logging.debug('Predicting and Calculating the Metrices for Prediction of Testing Dataset')
print('Predicting and Calculating the Metrices for Prediction of Testing Dataset')
prediction_test_rf = clf.predict(X_test)
r2_test_rf = r2_score(y_test, prediction_test_rf)
rms_test_rf = sqrt(mean_squared_error(y_test, prediction_test_rf))
mae_test_rf = mean_absolute_error(y_test,prediction_test_rf)
mape_test_rf = np.mean(np.abs((y_test - prediction_test_rf) / y_test)) * 100
       
logging.debug('Predicting and Calculating the Metrices for Prediction of Training Dataset')
print('Predicting and Calculating the Metrices for Prediction of Training Dataset')
prediction_train_rf = clf.predict(X_train)
r2_train_rf = r2_score(y_train, prediction_train_rf)
rms_train_rf = sqrt(mean_squared_error(y_train, prediction_train_rf))
mae_train_rf = mean_absolute_error(y_train,prediction_train_rf)
mape_train_rf = np.mean(np.abs((y_train - prediction_train_rf) / y_train)) * 100
  
logging.debug('Printing Metrices')
print('Printing Metrices')
print('r2_train_rf: ', float("{0:.2f}".format(r2_train_rf)))
print('r2_test_rf: ', float("{0:.2f}".format(r2_test_rf)))
print('rms_train_rf: ', float("{0:.2f}".format(rms_train_rf)))
print('rms_test_rf: ', float("{0:.2f}".format(rms_test_rf)))
print('mae_train_rf: ', float("{0:.2f}".format(mae_train_rf)))
print('mae_test_rf: ', float("{0:.2f}".format(mae_test_rf)))
print('mape_train_rf: ', float("{0:.2f}".format(mape_train_rf)))
print('mape_test_rf: ', float("{0:.2f}".format(mape_test_rf)))

logging.debug("Process Completed")
print("Process Completed")
