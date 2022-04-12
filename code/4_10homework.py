#实例7.1.2
from keras.models import Model
from keras import layers
from keras import Input
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
text_input = Input(shape = (None,),dtype = 'int32',name = 'text')
embedded_text = layers.Embedding(text_vocabulary_size,64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)
question_input = Input(shape = (None,),dtype='int32',name='question')
embedded_question = layers.Embedding(question_vocabulary_size,32)(question_input)
encoded_question = layers.LSTM(15)(embedded_question)
concatenated = layers.concatenate([encoded_text,encoded_question],axis = -1)
answer = layers.Dense(answer_vocabulary_size,activation = 'softmax')(concatenated)
model = Model([text_input,question_input],answer)
model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy',metrics = ['acc'])
import numpy as np
from keras.utils import np_utils
num_samples = 1000
max_length = 100
text = np.random.randint(1,text_vocabulary_size,size = (num_samples,max_length))
question = np.random.randint(1,question_vocabulary_size,size = (num_samples,max_length))
answers = np.random.randint(answer_vocabulary_size,size = (num_samples))
answers = np_utils.to_categorical(answers,answer_vocabulary_size)
model.fit([text,question],answers,epochs = 10,batch_size = 128)
model.fit({'text':text,'question':question},answers,epochs=10,batch_size = 128)
#第二题
from keras import Sequential
from keras.layers import Embedding,Flatten,Dense
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model,Sequential
#from tensorflow.keras.callbacks import EarlyStopping
max_features = 10000
maxlen = 500
batch_size = 32
(input_train,y_train),(input_test,y_test) = imdb.load_data(num_words=max_features)
input_train = sequence.pad_sequences(input_train,maxlen = maxlen)
input_test = sequence.pad_sequences(input_test,maxlen = maxlen)
model = Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics=['acc'])
#回调函数
early_stopping = EarlyStopping(monitor = 'val_accuracy',patience = 5,verbose = 1)
callbacks = [early_stopping]
history = model.fit(input_train,y_train,epochs = 10,batch_size = 128,validation_split=0.2,callbacks=callbacks)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.savefig('model_training_history')
plt.show()


#第三题
from keras import Sequential
from keras.layers import Embedding,Flatten,Dense
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.models import load_model,Sequential

max_features = 10000
maxlen = 500
batch_size = 32
(input_train,y_train),(input_test,y_test) = imdb.load_data(num_words=max_features)
input_train = sequence.pad_sequences(input_train,maxlen = maxlen)
input_test = sequence.pad_sequences(input_test,maxlen = maxlen)

model = Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics=['acc'])
early_stopping = EarlyStopping(monitor = 'val_accuracy',patience = 5,verbose = 1)
#加入Tensorboard可用的回调函数
callbacks = [
    TensorBoard(
        log_dir = 'my_log_dir',
        histogram_freq = 1,
        embeddings_freq = 1,
    )
]
history = model.fit(input_train,y_train,epochs = 10,batch_size = 128,validation_split=0.2,callbacks=callbacks)