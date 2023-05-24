#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np

data = pd.read_csv("file:///C:/Users/hp/Desktop/dataset.csv")
data


# In[54]:


data.shape


# In[55]:


data.mean(axis = 0)


# In[56]:


data.median(axis = 0)


# In[57]:


data


# In[58]:


data['ApplicantIncome'].mean()


# In[59]:


data['ApplicantIncome'].median()


# In[60]:


data['ApplicantIncome'].min()


# In[61]:


data['ApplicantIncome'].max()


# In[62]:


#Standard Derivation
data['ApplicantIncome'].std()


# In[63]:


#Varivation
data['ApplicantIncome'].var()


# In[64]:


data.groupby(['Gender']).count()


# In[65]:


data.groupby(['Education']).sum()


# In[72]:


q1 = np.median(data['ApplicantIncome'][len()])
q2 = np.median(data['ApplicantIncome'][75:])
iqr = q2 - q1
iqr


# In[75]:


data.skew()


# In[ ]:





# In[ ]:





# In[ ]:




