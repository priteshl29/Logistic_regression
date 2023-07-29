#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')


# In[3]:


titanic = pd.read_csv('E:\CONTANT/train.csv')


# In[4]:


print(titanic.head())


# In[5]:


print(titanic.info())


# In[6]:


print(titanic.isna().sum())


# In[7]:


print(titanic[['Age', 'SibSp', 'Parch', 'Fare']].describe())


# In[8]:


print(titanic['Survived'].value_counts())
print(titanic['Survived'].value_counts(normalize=True))


# In[9]:


fig, ax = plt.subplots(figsize=(5, 3))
sns.countplot(data=titanic, x='Survived')
plt.title('Survival count')
plt.show()


# In[10]:


print(titanic['Sex'].value_counts())


# In[11]:


sex_survived = pd.crosstab(titanic['Sex'], titanic['Survived'])
print(sex_survived)


# In[12]:


fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=titanic, x='Sex', hue='Survived')
plt.title('Survival count of female and male passengers')
plt.show()


# In[13]:


class_survived = pd.crosstab(titanic['Pclass'], titanic['Survived'])
print(class_survived)


# In[14]:


fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=titanic, x='Pclass', hue='Survived')
plt.title('Survival count and class')
plt.show()


# In[15]:


age_survived = pd.crosstab(titanic['Age'], titanic['Survived'])
print(age_survived)


# In[16]:


fig, ax = plt.subplots(figsize=(20, 10))
sns.histplot(data=titanic, x='Age', hue='Survived', bins=160, multiple='stack', ax=ax)
plt.title('Survival count and Age')
plt.show()


# In[17]:


fig, ax = plt.subplots(figsize=(17, 5))
sns.histplot(data=titanic, y='Parch', ax=ax, multiple='stack', hue='Survived', bins=10)
plt.title('Survival count for number of parents / children (Parch)')
plt.show()


# In[18]:


fig, ax = plt.subplots(figsize=(17, 5))
sns.histplot(data=titanic, y='SibSp', ax=ax, hue='Survived', multiple='stack', bins=10)
plt.title('Survival count for number of siblings / spouses (SibSp)')
plt.show()


# In[19]:


print(pd.crosstab(titanic['Parch'], titanic['Survived']))


# In[20]:


print(pd.crosstab(titanic['SibSp'], titanic['Survived']))


# In[21]:


print(pd.crosstab(titanic['Parch'], titanic['Pclass']))


# In[22]:


print(pd.crosstab(titanic['SibSp'], titanic['Pclass']))


# In[23]:


fig, ax = plt.subplots(figsize=(17, 5))
sns.histplot(data=titanic, y='Parch', ax=ax, multiple='stack', hue='Pclass', bins=10)
plt.title('Number of parents / children (Parch) per class')
plt.show()


# In[24]:


fig, ax = plt.subplots(figsize=(17, 5))
sns.histplot(data=titanic, y='SibSp', ax=ax, hue='Pclass', multiple='stack', bins=10)
plt.title('Number of siblings / spouses (SibSp) and class')
plt.show()


# In[25]:


print(titanic.groupby(['Parch', 'Pclass', 'Survived']).count()['PassengerId'])


# In[26]:


print(titanic.groupby(['SibSp', 'Pclass', 'Survived']).count()['PassengerId'])


# In[ ]:




