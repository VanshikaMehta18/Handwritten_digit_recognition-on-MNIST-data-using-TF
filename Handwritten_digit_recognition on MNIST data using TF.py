#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[24]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


mnist = tf.keras.datasets.mnist # 28x28 images of handwritten digits 0-9


# In[9]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[11]:


plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()


# In[12]:


print(x_train[0])
#we need to normalize this data 


# In[13]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[14]:


print(x_train[0])


# In[15]:


model = tf.keras.models.Sequential() #feed-forward nn
model.add(tf.keras.layers.Flatten()) #Flattening an image involves converting this multi-dimensional tensor into a one-dimensional tensor
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #softmax because its a prob distribution

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[16]:


model.fit(x_train, y_train, epochs=3)


# In[17]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


# In[18]:


model.save('num_reader')


# In[19]:


new_model = tf.keras.models.load_model('num_reader')


# In[23]:


predictions = new_model.predict([x_test])


# In[32]:


np.argmax(predictions[9])


# In[31]:


plt.imshow(x_test[9])


# In[ ]:




