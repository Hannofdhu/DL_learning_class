import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,layers,utils
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#build nn
model = Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(12,activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#data pre
data = load_iris()
print(data.feature_names)
print(data.target_names)
print(data.data[0])

X = data.data[:]
y = data.target[:]

train_x,test_x,train_y,test_y=train_test_split(X,y,\
                            train_size=0.8,test_size=0.2,\
                                    random_state=0)
mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_x = (train_x-mean)/std
test_x = (test_x-mean)/std
print(test_x[:5,])

train_y_one = utils.to_categorical(train_y,3)
test_y_one = utils.to_categorical(test_y,3)
print(train_y_one[:5,])

#model_compile
model.fit(train_x,train_y_one,epochs=50,batch_size=1,\
          verbose=2,validation_data=(test_x,test_y_one))
loss,accuracy = model.evaluate(test_x,test_y_one,verbose=2)
print(f'loss={loss},accuracy={accuracy}')
classes = model.predict(test_x,batch_size=1,verbose=2)
print('测试数：',len(classes))
print('分类概率:',classes)


