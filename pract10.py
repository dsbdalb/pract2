#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv('file:///C:/Users/hp/Documents/Adobe/pract2/iris.csv')
iris


# In[20]:


fig, axes = plt.subplots(2, 2, figsize=(16, 9))

sns.histplot(iris['sepal_length'], ax=axes[0, 0])

sns.histplot(iris['sepal_width'], ax=axes[0, 1])

sns.histplot(iris['petal_length'], ax=axes[1, 0])

sns.histplot(iris['petal_width'], ax=axes[1, 1])


# In[19]:


fig, axes = plt.subplots(2, 2, figsize=(16, 9))

sns.boxplot(x='species', y='sepal_length', data=iris, ax=axes[0, 0])

sns.boxplot(x='species', y='sepal_width', data=iris, ax=axes[0, 1])

sns.boxplot(x='species', y='petal_length', data=iris, ax=axes[1, 0])

sns.boxplot(x='species', y='petal_width', data=iris, ax=axes[1, 1])


# In[ ]:




