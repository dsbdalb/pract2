#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd

data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj', 'Ravi', 'Natasha', 'Riya'],
        'Age': [17, 17, 18, 17, 18, 17, 17],
        'Gender': ['M', 'F', 'M', 'M', 'M', 'F', 'F'],
        'Marks': [90, 76, 'NaN', 74, 65, 'NaN', 71]}

df = pd.DataFrame(data)
df


# In[14]:


c = avg = 0
for ele in df['Marks']:
    if str(ele).isnumeric():
        c += 1
        avg += ele
avg /= c

df = df.replace(to_replace='NaN', value=avg)
df


# In[15]:


df['Gender'] = df['Gender'].map({'M': 0,'F': 1}).astype(float)
df


# In[16]:


df = df[df['Marks'] >= 75]
df = df.drop(['Age'], axis=1)
df


# In[18]:


details = pd.DataFrame({ 'ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                        'NAME': ['Jagroop', 'Praveen', 'Harjot', 'Pooja', 'Rahul', 'Nikita', 'Saurabh', 'Ayush', 'Dolly', "Mohit"],
                        'BRANCH': ['CSE', 'CSE', 'CSE', 'CSE', 'CSE','CSE', 'CSE', 'CSE', 'CSE', 'CSE']})
details


# In[21]:


fees_status = pd.DataFrame({'ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                            'PENDING': ['5000', '250', 'NIL', '9000', '15000', 'NIL', '4500', '1800', '250', 'NIL']})
fees_status


# In[23]:


pd.merge(details, fees_status, on='ID')


# In[24]:


car_selling_data = {'Brand': ['Maruti', 'Maruti', 'Maruti', 'Maruti', 'Hyundai', 'Hyundai',
'Toyota', 'Mahindra', 'Mahindra', 'Ford', 'Toyota', 'Ford'],
                    'Year': [2010, 2011, 2009, 2013, 2010, 2011, 2011, 2010, 2013, 2010, 2010, 2011],
                    'Sold': [6, 7, 9, 8, 3, 5, 2, 8, 7, 2, 4, 2]}
df = pd.DataFrame(car_selling_data)
df


# In[28]:


car_selling_data = {'Brand': ['Maruti', 'Maruti', 'Maruti', 'Maruti', 'Hyundai', 'Hyundai', 'Toyota', 'Mahindra', 'Mahindra',
'Ford', 'Toyota', 'Ford'],
                    'Year': [2010, 2011, 2009, 2013, 2010, 2011, 2011, 2010, 2013, 2010, 2010, 2011],
                    'Sold': [6, 7, 9, 8, 3, 5, 2, 8, 7, 2, 4, 2]}
df = pd.DataFrame(car_selling_data)
grouped = df.groupby('Year')
grouped.get_group(2010)


# In[29]:


student_data = {'Name': ['Amit', 'Praveen', 'Jagroop', 'Rahul', 'Vishal', 'Suraj', 'Rishab', 'Satyapal', 'Amit', 'Rahul', 'Praveen', 'Amit'],
                'Roll_no': [23, 54, 29, 36, 59, 38, 12, 45, 34, 36, 54, 23],
                'Email': ['xxxx@gmail.com', 'xxxxxx@gmail.com', 'xxxxxx@gmail.com', 'xx@gmail.com', 'xxxx@gmail.com', 'xxxxx@gmail.com', 'xxxxx@gmail.com', 'xxxxx@gmail.com', 'xxxxx@gmail.com', 'xxxxxx@gmail.com', 'xxxxxxxxxx@gmail.com', 'xxxxxxxxxx@gmail.com']}
df = pd.DataFrame(student_data)
df


# In[33]:


non_duplicate = df[~df.duplicated('Roll_no')]
non_duplicate


# In[ ]:




