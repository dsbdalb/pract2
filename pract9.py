#!/usr/bin/env python
# coding: utf-8

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

titanic = pd.read_csv('file:///C:/Users/hp/Documents/Adobe/pract8/titanic.csv')
titanic.head()


# In[15]:


sns.boxplot(x='sex', y='age', data=titanic)


# In[16]:


sns.boxplot(x='sex', y='age', data=titanic, hue='survived')
plt.title("Box Plot for Survived wrt to age")
plt.show()


# In[ ]:




