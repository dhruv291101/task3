#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.datasets import mnist


# In[ ]:


dataset = mnist.load_data('mymnist.db')


# In[ ]:


len(dataset)


# In[ ]:


train , test = dataset


# In[ ]:


len(train)


# In[ ]:


X_train , y_train = train


# In[ ]:


X_train.shape


# In[ ]:


X_test , y_test = test


# In[ ]:


X_test.shape


# In[ ]:


img1 = X_train[7]


# In[ ]:


img1.shape


# In[ ]:


import cv2


# In[ ]:


img1_label = y_train[7]


# In[ ]:


img1_label


# In[ ]:


img1.shape


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.imshow(img1 , cmap='gray')


# In[ ]:


img1.shape


# In[ ]:


img1_1d = img1.reshape(28*28)


# In[ ]:


img1_1d.shape


# In[ ]:


X_train.shape


# In[ ]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[ ]:


X_train_1d.shape


# In[ ]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


from keras.utils.np_utils import to_categorical


# In[ ]:


y_train_cat = to_categorical(y_train)


# In[ ]:


y_train_cat


# In[ ]:


y_train_cat[7]


# In[ ]:


from keras.models import Sequential


# In[ ]:


from keras.layers import Dense


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Dense(units=512, input_dim=28*28, activation='relu'))


# In[ ]:


model.summary()


# In[ ]:


model.add(Dense(units=256, activation='relu'))


# In[ ]:


model.add(Dense(units=128, activation='relu'))


# In[ ]:


model.add(Dense(units=32, activation='relu'))


# In[ ]:


model.summary()


# In[ ]:


model.add(Dense(units=10, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


from keras.optimizers import RMSprop


# In[ ]:


model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[ ]:


h = model.fit(X_train, y_train_cat, epochs=20)


# In[ ]:


plt.imshow(X_test[0])


# In[ ]:


y_test[0]


# In[ ]:


model.predict(X_test[0])


# In[ ]:


test_img = X_test[0].reshape(28*28)


# In[ ]:


test_img.shape


# In[ ]:


model.predict(test_img)


# In[ ]:




