from tensorflow.keras import models,layers,utils
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#一定程度上抑制了warning
import tensorflow as tf
tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
#build nn
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#数据处理
data = pd.read_csv('/Users/hann/Documents/ju_code/DL_learning_class/data/iris.data.txt',header=None)
data.columns = ['sepal length','sepal width','petal length','petal width','class']
print(data.iloc[0:20,:])
# data = load_iris()
# print(data.feature_names)
# print(data.target_names)
# print(data.data[0])
# X = data.data[:]
# y = data.target[:]
X = data.iloc[:,0:4].values.astype(float)
data.loc[data['class']=='Iris-setosa','class']=0
data.loc[data['class']=='Iris-versicolor','class']=1
data.loc[data['class']=='Iris-virginica','class']=2
y = data.iloc[:,4].values.astype(int)
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
model.fit(train_x,train_y_one,epochs=20,batch_size=1,verbose=2,validation_data=(test_x,test_y_one))
model.evaluate(test_x,test_y_one,verbose=2)
#print(f'loss={loss},accuracy={accuracy}')
#loss,accuracy =
classes = model.predict(test_x,batch_size=1,verbose=2)
print('测试数：',len(classes))
print('分类概率:',classes)


