#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statistics


# In[20]:


df = pd.read_excel('df.xlsx', sheet_name='Sheet1')
df20 = pd.read_excel('df.xlsx', sheet_name='YearF2020')
features = ['DayOrder', 'Season', 'DayOfTheWeek', 'Period','Route']
r1=len(df)


# In[21]:


print("missing data :",df.isnull().sum())


# In[22]:


print("incosistent data :")
print((df['DayOrder'] <= 0).sum())
print((df['Season'] <= 0).sum())
print((df['DayOfTheWeek'] <= 0).sum())
print((df['Period'] <= 0).sum())
print((df['Tij'] <= 0).sum())


# In[23]:


index_Day = df[ df['DayOfTheWeek'] <= 0 ].index
index_T = df[ df['Tij'] <= 0 ].index
df.drop(df[ df['DayOfTheWeek'] <= 0 ].index, inplace = True) 
df.drop(df[ df['Tij'] <= 0 ].index, inplace = True) 


# In[24]:


print("% Rows dropped:",(r1-len(df))*100/r1)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(df[features], df['Tij'] , test_size=0.20,random_state=1)


# In[8]:



rg = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 200, num = 5)]
max_depth.append(None)
min_samples_split = [2,3,4 ,5,6,7,8,9,10]
min_samples_leaf = [1, 2, 3, 4,5]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rg, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=0, random_state=1, n_jobs = -1)
rf_random.fit(X_train,y_train)


# In[ ]:


rf_random.best_params_


# In[26]:


rg = RandomForestRegressor(n_estimators= 773,
 min_samples_split= 8,
 min_samples_leaf=  5,
 max_features= 'sqrt',
 max_depth= None,
 bootstrap= True,
    random_state=1)
rg.fit(X_train,y_train)
y_pred = rg.predict(X_test)


# In[27]:


finaldf=X_test.copy()
finaldf['True']=y_test
finaldf['Pred']=y_pred


# In[29]:


finaldf=finaldf.sort_values(by=['DayOrder'])
plt.figure(figsize=(40,20))
plt.plot( 'DayOrder', 'True', data=finaldf, marker='o', linewidth=2, label='True')
plt.plot( 'DayOrder', 'Pred', data=finaldf, marker='o', linewidth=2, label='Predicted')
plt.xlabel("Tij")
plt.ylabel("Day Order")
plt.legend()
plt.savefig('rf_test.png')


# In[30]:


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle('Comparison between Pred and True')
sns.boxplot(ax=ax1, data=finaldf, y='True')
ax1.set_ylim(0,25)
sns.boxplot(ax=ax2, data=finaldf, y='Pred')
ax2.set_ylim(0,25)
plt.savefig('test.png')


# In[14]:


finaldf[['True','Pred']].describe()


# In[31]:


fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(18, 10))
fig.suptitle('Comparison between Pred and True')
ax1.hist(finaldf['True'])
ax1.set_ylim(0,200)
ax1.set_xlim(0,25)
ax1.set_title("True")
ax2.hist(finaldf['Pred'])
ax2.set_ylim(0,200)
ax2.set_xlim(0,25)
ax2.set_title("Pred")
plt.savefig('comp_test.png')


# In[16]:


max_pred=max(finaldf['Pred'])
perc = (len(df[df['Tij']>max_pred])/len(df['Tij']))*100
print("percentage of the data above %s = %s" %(max_pred,perc))


# In[32]:


residuals = y_train-rg.predict(X_train)
plt.figure(figsize=(10, 10))
plt.title('Residual')
plt.hist(residuals)
plt.savefig('resi_rf.png')


# In[35]:


sm.graphics.tsa.plot_acf(residuals, lags=20)
plt.savefig('aut_rf.png')
plt.show()


# In[19]:


def mape(y_test,y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_pred)) * 100
MAPE = mape(y_test,y_pred)
MAE = mean_absolute_error(y_test, y_pred)
ME = statistics.mean(y_test-y_pred)


# In[20]:


print("MAPE = %s \nMAE = %s" %(MAPE,MAE))
print("ME = %s" %(ME))


# In[21]:


forecaste=rg.predict(df20)
df20['Tij_predicted']=forecaste


# from openpyxl import load_workbook
# excel_dir = r"C:\Users\Jaloud\Math_VR\For_data.xlsx"
# 
# 
# book = load_workbook(excel_dir)
# writer = pd.ExcelWriter(excel_dir, engine = 'openpyxl')
# writer.book = book
# 
# 
# df20.to_excel(writer, sheet_name = 'Random_forest')
# writer.save()
# writer.close()
