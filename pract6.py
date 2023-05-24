#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[17]:


data = pd.read_csv('file:///C:/Users/hp/Desktop/iris.csv')


# In[18]:


X = data.drop('species', axis=1)
y = data['species']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


nb_classifier = GaussianNB()


# In[21]:


nb_classifier.fit(X_train, y_train)


# In[22]:


y_pred = nb_classifier.predict(X_test)


# In[23]:


confusion_mat = confusion_matrix(y_test, y_pred)
tn, fp, fn = confusion_mat[0, 0], confusion_mat[0, 1], confusion_mat[1, 0]
tp = confusion_mat[1, 1] + confusion_mat[2, 2]


# In[24]:


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [True, True])
cm_display.plot()
plt.show()


# In[25]:


accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')


# In[26]:


print("Confusion Matrix:")
print(confusion_mat)
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)


# In[ ]:




