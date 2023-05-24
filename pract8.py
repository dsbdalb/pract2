#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('file:///C:/Users/hp/Documents/Adobe/pract8/titanic.csv')
dataset.head()


# # The Dist Plot

# In[11]:


sns.distplot(dataset['fare'])


# In[12]:


sns.distplot(dataset['fare'], kde=False)


# In[13]:


sns.distplot(dataset['fare'], kde=False, bins=10)


# # The Joint Plot

# In[14]:


sns.jointplot(x='age', y='fare', data=dataset)


# In[15]:


sns.jointplot(x='age', y='fare', data=dataset, kind='hex')


# # The Pair Plot

# In[16]:


sns.pairplot(dataset)


# In[17]:


dataset = dataset.dropna()


# In[18]:


sns.pairplot(dataset, hue='sex')


# # The Rug Plot

# In[19]:


sns.rugplot(dataset['fare'])


# # Categorical Plots

# # The Bar Plot

# In[22]:


sns.barplot(x='sex', y='age', data=dataset)


# In[23]:


sns.barplot(x='sex', y='age', data=dataset, estimator=np.std)


# # The Count Plot

# In[26]:


sns.countplot(x='sex', data=dataset)


# # The Box Plot

# In[27]:


sns.boxplot(x='sex', y='age', data=dataset)


# In[28]:


sns.boxplot(x='sex', y='age', data=dataset, hue="survived")


# # The Violin Plot

# In[29]:


sns.violinplot(x='sex', y='age', data=dataset)


# In[30]:


sns.violinplot(x='sex', y='age', data=dataset, hue='survived')


# In[31]:


sns.violinplot(x='sex', y='age', data=dataset, hue='survived', split=True)


# # The Strip Plot

# In[33]:


sns.stripplot(x='sex', y='age', data=dataset)


# In[34]:


sns.stripplot(x='sex', y='age', data=dataset, jitter=True)


# In[35]:


sns.stripplot(x='sex', y='age', data=dataset, jitter=True, hue='survived')


# In[37]:


sns.stripplot(x='sex', y='age', data=dataset, jitter=True, hue='survived')


# # The Swarm Plot

# In[38]:


sns.swarmplot(x='sex', y='age', data=dataset)


# In[39]:


sns.swarmplot(x='sex', y='age', data=dataset, hue='survived')


# In[41]:


sns.swarmplot(x='sex', y='age', data=dataset, hue='survived')


# # Combining Swarm and Violin Plots

# In[42]:


sns.violinplot(x='sex', y='age', data=dataset)
sns.swarmplot(x='sex', y='age', data=dataset, color='black')


# In[ ]:




