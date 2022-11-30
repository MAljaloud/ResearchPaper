#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy import stats


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


# ## No need to go to matrix completion or another technique to fill the unconsistent/Null data since they are very small portion of the data

# In[10]:


df17_temp=df17.copy()
df18_temp=df18.copy()
df19_temp=df19.copy()
df17_temp['Date']= 2017
df18_temp['Date']= 2018
df19_temp['Date']= 2019
frames1 = [df17_temp,df18_temp,df19_temp]
df_temp = pd.concat(frames1)
df_temp.reset_index(inplace = True)
df_temp=df_temp.drop(['index'],axis=1)


# In[11]:


sns.pairplot(data=df_temp, hue="Date",palette='Dark2_r')
plt.savefig('pairplot.png')


# ## The lowest diagonaol elements shows the distribution of Tij for the 3 years, the year 2018 shows perfect Normal distribution (Normal). To able to concloude that we need Statistical tests

#                             # H0 = 2017,2019 time follow the distribution as 2018 time
#                             # H1 = The above statment is false and they are different 
#                             alpha = 0.05

# In[12]:


print(stats.ks_2samp(df17['Tij'], df18['Tij']))
print(stats.ks_2samp(df18['Tij'], df19['Tij']))


# ## The result gave us the same conclusion as the figure, even though 2017 p-value is low. I believe that it occured because of because of noise/missing data

# ## ------------------------------------------------------------------------------------------------------------------------------

# ## The following section is just plots to underastand the data more

# In[13]:


frames = [df17, df18,df19]
df = pd.concat(frames)
df.reset_index(inplace = True)
df=df.drop(['index'],axis=1)


# In[14]:


ax = sns.boxplot(x="Season", y="Tij", hue="Date",
                 data=df_temp, palette="Set3")
sns.set(rc={'figure.figsize':(30,10)})


# In[15]:


df17=df17.sort_values(by=['DayOrder'])
df18=df18.sort_values(by=['DayOrder'])
df19=df19.sort_values(by=['DayOrder'])
fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(df17['DayOrder'],df17['Tij'])
axs[1].plot(df18['DayOrder'],df18['Tij'])
axs[2].plot(df19['DayOrder'],df19['Tij'])


# In[16]:


df17_season1 = df17[(df17['Season'] == 1)]
df17_season2 = df17[(df17['Season'] == 2)]
df17_season3 = df17[(df17['Season'] == 3)]
df18_season1 = df18[(df18['Season'] == 1)]
df18_season2 = df18[(df18['Season'] == 2)]
df18_season3 = df18[(df18['Season'] == 3)]
df19_season1 = df19[(df19['Season'] == 1)]
df19_season2 = df19[(df19['Season'] == 2)]
df19_season3 = df19[(df19['Season'] == 3)]
df_all1  = df[(df['Season'] == 1)]
df_all2  = df[(df['Season'] == 2)]
df_all3  = df[(df['Season'] == 3)]


# In[17]:


df_all1=df_all1.sort_values(by=['DayOrder'])
df_all2=df_all2.sort_values(by=['DayOrder'])
df_all3=df_all3.sort_values(by=['DayOrder'])
plt.plot( 'DayOrder', 'Tij', data=df_all1, marker='o', linewidth=2)
plt.plot( 'DayOrder', 'Tij', data=df_all2, marker='', color='green', linewidth=2)
plt.plot( 'DayOrder', 'Tij', data=df_all3, marker='', color='red', linewidth=2)
plt.hlines(df_all1['Tij'].mean(), df_all1['DayOrder'].min(), df_all1['DayOrder'].max(), color='black')
plt.hlines(df_all2['Tij'].mean(), df_all2['DayOrder'].min(), df_all2['DayOrder'].max(), color='black')
plt.hlines(df_all3['Tij'].mean(), df_all3['DayOrder'].min(), df_all3['DayOrder'].max(), color='black')
plt.legend()
plt.show()


# In[18]:


ax = sns.boxplot(x=df_all1['DayOfTheWeek'], y=df_all1['Tij'])


# In[19]:


ax = sns.boxplot(x=df_all1['Period'], y=df_all1['Tij'])


# In[20]:


ax = sns.boxplot(x=df_all2['DayOfTheWeek'], y=df_all2['Tij'])


# In[21]:


ax = sns.boxplot(x=df_all2['Period'], y=df_all2['Tij'])


# In[23]:


ax = sns.boxplot(x=df_all3['DayOfTheWeek'], y=df_all3['Tij'])


# In[24]:


ax = sns.boxplot(x=df_all3['Period'], y=df_all3['Tij'])

