#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


df = pd.read_csv('foreveralone.csv')
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.columns


# ## Feature Selection

# According to case study we can consider some important featuers 
# 
# ['gender', 'sexuallity', 'friends', 'age', 'income', 'bodyweight', 'virgin', 'social_fear', 'depressed']
# 
# 
# target variable = 'attempt_suicide'

# In[6]:


cols = ['gender', 'sexuallity', 'friends', 'age', 'income', 'bodyweight', 'virgin', 'social_fear', 'employment', 'depressed', 'attempt_suicide']


# In[7]:


for i in cols:
    print('-----------------------------------')
    print(i, ':')
    print(df[i].value_counts())
    #print('Unique Values :', df[i].unique())
    print('Number of Unique values:', len(df[i].unique()))
    print('\n')


# In[8]:


df1 = df[cols]


# In[9]:


df1


# In[10]:


df1.isna().sum()


# In[11]:


cat_cols = ['gender', 'sexuallity', 'bodyweight', 'virgin', 'social_fear', 'depressed', 'employment', 'attempt_suicide']


# In[12]:


for i in cat_cols:
    print('-----------------------------------')
    print(i, ':')
    print(df1[i].value_counts())
    #print('Unique Values :', df[i].unique())
    print('Number of Unique values:', len(df[i].unique()))
    print('\n')


# In[13]:


df1['gender'].replace('Transgender male', 'Male', inplace=True)

df1['gender'].replace('Transgender female', 'Female', inplace=True)


# In[14]:


df1.nunique()


# In[15]:


df1


# In[16]:


df1['income']


# In[17]:


df1['income'].value_counts()


# In[18]:


df1


# In[19]:


df1.info()


# In[20]:


df1.columns


# In[21]:


df1.info()


# In[22]:


df1.describe()


# In[23]:


sns.countplot(df1['gender'])


# In[24]:


sns.countplot(df1['sexuallity'])


# In[25]:


sns.countplot(df1['attempt_suicide'])


# In[26]:


sns.countplot(df1['bodyweight'])


# In[27]:


sns.countplot(df1['virgin'])


# In[28]:


sns.countplot(df1['social_fear'])


# In[29]:


sns.countplot(df1['depressed'])


# In[30]:


df1.corr()


# In[31]:


plt.figure(figsize=(14,14))
sns.heatmap(df1.corr(), annot=True, fmt='.0%')


# In[32]:


df1.tail(20)


# In[33]:


for column in df1.columns:
    if df1[column].dtype == 'object':
        df1[column] = LabelEncoder().fit_transform(df1[column])


# In[34]:


final_df = df1


# In[35]:


X = final_df.drop('attempt_suicide', axis=1)
y = final_df['attempt_suicide']


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=92)


# In[37]:


model = LogisticRegression()


# In[38]:


model.fit(X_train, y_train)


# In[39]:


model.score(X_train, y_train)


# In[40]:


model.score(X_test, y_test)


# In[41]:


rf = RandomForestClassifier()


# In[42]:


rf.fit(X_train, y_train)


# In[43]:


print('Confustion Matrix : \n\n', confusion_matrix(y_test,  rf.predict(X_test)))
print('\n Accuracy Score : ',   accuracy_score(y_test,  rf.predict(X_test)))
print('\n Classification Report : \n \n',classification_report(y_test, rf.predict(X_test)))


# In[44]:


rf.predict([[1,2,3.0,28,10,0,1,1,2,1]])


# In[45]:


final_df[final_df['attempt_suicide'] == 1]


# In[46]:


rf.predict([[1,2,5.0,8,1,2,1,1,2,1]])


# In[47]:


rf.feature_importances_


# In[48]:


for i in range(len(final_df.columns)-1):
    print(f"Feature importance of {final_df.columns[i]}  :   {rf.feature_importances_[i]}")


# In[49]:


# saving the model to the local file system
filename = 'model.pkl'
joblib.dump(rf, open(filename, 'wb'))


# In[50]:


joblib.load(open('model.pkl','rb'))


# In[ ]:




