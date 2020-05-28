#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()


# In[3]:


def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)
        
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[4]:


l_length = [3, 4, 3.5, 5.5, 2, 3, 2, 1]
l_width = [1.5, 1.5, .5, 1, 1, 1, .5, 1]
color = [0, 0, 0, 0, 1, 1, 1, 1]
data = []
prediction = []


# In[5]:


i = 0
for l in l_length:
    w = l_width[i]
    c = color[i]
    data.append([l, w, c])
    if c:
        plt.scatter(l, w, color='red')
    else:
        plt.scatter(l, w, color='blue') 
    i += 1


# In[6]:


data_f = pd.DataFrame(data, columns=['Length', 'Width', 'Color'])


# In[7]:


for characteristics in data:
    l = characteristics[0]
    w = characteristics[1]
    prediction.append(NN(l, w, w1, w2, b))


# In[8]:


data_f['Prediction'] = prediction


# In[9]:


print(data_f)


# In[ ]:




