from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 20
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
print(x_train[:1,])

x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = sequence.pad_sequences(x_test,maxlen=maxlen)
print(x_train[:1,])

from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding
model = Sequential()
model.add(Embedding(10000,8,input_length = 20))
model.add(Flatten())
model.add(Dense(1,activation = 'sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss = 'binary_crossentropy',metrics = ['acc'])
history = model.fit(x_train,y_train,epochs=10,batch_size = 32,validation_split=0.2)


max_features = 10000
maxlen = 60
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
print(x_train[:1,])

x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = sequence.pad_sequences(x_test,maxlen=maxlen)
print(x_train[:1,])
model = Sequential()
model.add(Embedding(10000,8,input_length = 60))
model.add(Flatten())
model.add(Dense(1,activation = 'sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss = 'binary_crossentropy',metrics = ['acc'])
history = model.fit(x_train,y_train,epochs=10,batch_size = 32,validation_split=0.2)


max_features = 10000
maxlen = 60
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
print(x_train[:1,])

x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = sequence.pad_sequences(x_test,maxlen=maxlen)
print(x_train[:1,])
model = Sequential()
model.add(Embedding(10000,16,input_length = 60))
model.add(Flatten())
model.add(Dense(1,activation = 'sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss = 'binary_crossentropy',metrics = ['acc'])
history = model.fit(x_train,y_train,epochs=10,batch_size = 32,validation_split=0.2)

