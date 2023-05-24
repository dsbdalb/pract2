#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


boston = pd.read_csv('file:///C:/Users/hp/Documents/Adobe/pract4/boston.csv')
boston.head()


# In[15]:


boston.shape


# In[53]:


# CRIM: Per capita crime rate by town
# ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
# INDUS: Proportion of non-retail business acres per town
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX: Nitric oxide concentration (parts per 10 million)
# RM: Average number of rooms per dwelling
# AGE: Proportion of owner-occupied units built prior to 1940
# DIS: Weighted distances to five Boston employment centers
# RAD: Index of accessibility to radial highways
# TAX: Full-value property tax rate per $10,000
# PTRATIO: Pupil-teacher ratio by town
# B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
# LSTAT: Percentage of lower status of the population
# MEDV: Median value of owner-occupied homes in $1000s


# In[16]:


print(boston.keys())


# In[17]:


boston.isnull().sum()


# # Exploratory Data Analysis
# 

# In[19]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['medv'], bins=30)
plt.show()


# In[22]:


correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


# In[ ]:


# To fit a linear regression model, we select those features which have a high correlation with our target variable MEDV.
# By looking at the correlation matrix we can see that RM has a strong positive correlation with MEDV (0.7)
# where as LSTAT has a high negative correlation with MEDV(-0.74).
# An important point in selecting features for a linear regression model is to check for multi-co-linearity.
# The features RAD, TAX have a correlation of 0.91. These feature pairs are strongly correlated to each other.
# We should not select both these features together for training the model. Check this for an explanation.
# Same goes for the features DIS and AGE which have a correlation of -0.75.
# Based on the above observations we will RM and LSTAT as our features.
# Using a scatter plot let’s see how these features vary with MEDV.


# In[54]:


plt.figure(figsize=(20, 5))

features = ['lstat', 'rm']
target = boston['medv']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# In[25]:


# The prices increase as the value of RM increases linearly. There are few outliers and the data seems to be capped at 50.
# The prices tend to decrease with an increase in LSTAT. Though it doesn’t look to be following exactly a linear line.


# # Preparing the data for training the model

# In[28]:


X = pd.DataFrame(np.c_[boston['lstat'], boston['rm']], columns = ['lstat','rm'])
Y = boston['medv']


# # Splitting the data into training and testing sets
# 

# In[27]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# # Training and testing the model
# 

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# # Model evaluation

# In[44]:


y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




