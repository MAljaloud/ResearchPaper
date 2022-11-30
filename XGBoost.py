#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error


# In[2]:


df17 = pd.read_excel('Project_data.xlsx', sheet_name='Year2017')
df18 = pd.read_excel('Project_data.xlsx', sheet_name='Year2018')
df19 = pd.read_excel('Project_data.xlsx', sheet_name='Year2019')
df20 = pd.read_excel('Project_data.xlsx', sheet_name='YearF2020')
r1=len(df17)
r2=len(df18)
r3=len(df19)
features = ['DayOrder', 'Season', 'DayOfTheWeek', 'Period']
print(r1)
print(r2)
print(r3)


# In[3]:


print("2017 missing data :",df17.isnull().sum())
print("2018 missing data :",df18.isnull().sum())
print("2019 missing data :",df19.isnull().sum())


# In[4]:


print("2017 incosistent data :")
print((df17['DayOrder'] <= 0).sum())
print((df17['Season'] <= 0).sum())
print((df17['DayOfTheWeek'] <= 0).sum())
print((df17['Period'] <= 0).sum())
print((df17['Tij'] <= 0).sum())


# In[5]:


print("2018 incosistent data :")
print((df18['DayOrder'] <= 0).sum())
print((df18['Season'] <= 0).sum())
print((df18['DayOfTheWeek'] <= 0).sum())
print((df18['Period'] <= 0).sum())
print((df18['Tij'] <= 0).sum())


# In[6]:


print("2019 incosistent data :")
print((df19['DayOrder'] <= 0).sum())
print((df19['Season'] <= 0).sum())
print((df19['DayOfTheWeek'] <= 0).sum())
print((df19['Period'] <= 0).sum())
print((df19['Tij'] <= 0).sum())


# In[7]:


index_2018Day = df18[ df18['DayOfTheWeek'] <= 0 ].index
index_2018T = df18[ df18['Tij'] <= 0 ].index
index_2019Day = df19[ df19['DayOfTheWeek'] <= 0 ].index
index_2019T = df19[ df19['Tij'] <= 0 ].index


# In[8]:


df18.drop(df18[ df18['DayOfTheWeek'] <= 0 ].index, inplace = True) 
df18.drop(df18[ df18['Tij'] <= 0 ].index, inplace = True) 
df19.drop(df19[ df19['DayOfTheWeek'] <= 0 ].index, inplace = True)
df19.drop(df19[ df19['DayOfTheWeek'] <= 0 ].index, inplace = True)


# In[9]:


print("% Rows dropped from 2018:",(r2-len(df18))*100/r2)
print("% Rows dropped from 2019:",(r3-len(df19))*100/r3)


# In[10]:


frames = [df17, df18,df19]
df = pd.concat(frames)
df.reset_index(inplace = True)
df=df.drop(['index'],axis=1)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(df[features], df['Tij'], test_size=0.20,random_state=1)


# In[12]:


model_rand = xgb.XGBRegressor()


n_estimators = [100,250,300,400,500]
max_depth = [1,2,3,4,5]
booster=['gbtree','gblinear']
learning_rate=[0.10,0.15,0.20,0.25,0.30,0.35]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]
num_boost_round=[10,15,20]
reg_alpha=[int(x) for x in np.linspace(start = 0, stop = 1000, num = 50)]
reg_lambda=[int(x) for x in np.linspace(start = 0, stop = 1000, num = 50)]
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score,
    'num_boost_round':num_boost_round,
    'reg_alpha':reg_alpha,
    'reg_lambda':reg_lambda  }


# In[13]:


random_cv = RandomizedSearchCV(estimator=model_rand,
            param_distributions=hyperparameter_grid,
            cv=10, n_iter=50,
             n_jobs =-1,
            verbose = 2, 
            return_train_score = True,
            random_state=1)


# In[14]:


random_cv.fit(X_train,y_train)


# In[15]:


random_cv.best_estimator_


# In[16]:


model = xgb.XGBRegressor(base_score=0.75, learning_rate=0.15, max_depth=4,
             min_child_weight=2, n_estimators=500, num_boost_round=10,
             reg_alpha=20, reg_lambda=1000).fit(X_train,y_train)
y_pred = model.predict(X_test)
finaldf=X_test.copy()
finaldf['True']=y_test
finaldf['Pred']=y_pred


# In[17]:


mean_absolute_error(y_test, y_pred)


# In[18]:


finaldf=X_test.copy()
finaldf['True']=y_test
finaldf['Pred']=y_pred


# In[19]:


finaldf=finaldf.sort_values(by=['DayOrder'])
plt.figure(figsize=(10,5))
plt.plot( 'DayOrder', 'True', data=finaldf, marker='o', linewidth=2, label='True')
plt.plot( 'DayOrder', 'Pred', data=finaldf, marker='o', linewidth=2, label='Predicted')
plt.xlabel("Tij")
plt.ylabel("Day Order");
plt.legend()
plt.savefig('gb_test.png')


# In[22]:


def mape(y_test,y_pred): 
    y_test,y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_pred)) * 100


# In[23]:


print(mape(y_test, y_pred))


# In[20]:


Train=X_train.copy()
Train['True']=y_train
Train['Pred']=model.predict(X_train)


# In[21]:


Train=Train.sort_values(by=['DayOrder'])
plt.figure(figsize=(17,5))
plt.plot( 'DayOrder', 'True', data=Train, marker='o', linewidth=2, label='True')
plt.plot( 'DayOrder', 'Pred', data=Train, marker='o', linewidth=2, label='Predicted')
plt.xlabel("Tij")
plt.ylabel("Day Order")
plt.legend()
plt.savefig('gb_train.png')


# In[134]:


forecaste=model.predict(df20)
df20['Tij_predicted']=forecaste


# In[136]:


from openpyxl import load_workbook
excel_dir = r"C:\Users\Jaloud\Desktop\Projects\Math405 Project\ForecastedTime.xlsx"

path = r"C:\Users\Jaloud\Desktop\Projects\Math405 Project\ForecastedTime.xlsx"

book = load_workbook(path)
writer = pd.ExcelWriter(path, engine = 'openpyxl')
writer.book = book


df20.to_excel(writer, sheet_name = 'XGB_forecast')
writer.save()
writer.close()


# In[ ]:




