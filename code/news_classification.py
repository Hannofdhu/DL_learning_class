"""
无论问题是什么， rmsprop 优化器通常都是足够好的选择。
尝试使用 不同的 batch_size
尝试设置达到一定的验证精度，停止训练（提示：用 if 语句）此处使用的是keras的早停机制
"""
from tensorflow import  keras
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import  to_categorical
from tensorflow.keras import optimizers,losses,metrics,callbacks
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import copy

#图像绘制
def draw_display(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1,len(loss_values)+1)
    plt.plot(epochs, loss_values, 'bo' , label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
#序列向量化
def vectorize_sequences(sequences,dimention=10000):
    results = np.zeros((len(sequences),dimention))
    for i,sequence in enumerate(sequences):
        #矩阵对饮i和sequence的地方为1
        results[i,sequence] = 1.
    return results

#主函数
def main(args):
    #build model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(args.neuron,activation=args.activation,input_shape=(10000,)))
    model.add(keras.layers.Dense(args.neuron, activation=args.activation))
    model.add(keras.layers.Dropout(args.drop_out))
    model.add(keras.layers.Dense(46,activation='softmax'))
    model.compile(loss=args.loss,optimizer=args.optimizer,metrics=['accuracy'])

    #data_processing
    (train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)
    print(train_data[0])
    print(train_labels[0])
    #句子的长度
    print(len(train_data[0]),len(train_data[1]))
        #data convert
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    print(x_train[0])
        #独热编码
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]
    #model_fit
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss',verbose=1, mode='min')#, patience=10
    history = model.fit(partial_x_train,partial_y_train,epochs=args.epochs,\
          batch_size=args.batch_size,verbose=2,validation_data=(x_val,y_val),callbacks=earlyStopping)
    draw_display(history)
    #model_evaluate
    loss,accuracy = model.evaluate(x_test,one_hot_test_labels)
    print(f'loss={loss},accuracy={accuracy}')

    #model_prediction
    classes = model.predict(x_test,batch_size=args.batch_size,verbose=2)
    print('测试数：',len(classes))
    print('分类概率:',classes)
    #打印随机分类效果
    test_labels_copy = copy.copy(test_labels)
    np.random.shuffle(test_labels_copy)
    random_acc = float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels)
    print(f'随机分类正确率：{random_acc}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #config
    parser.add_argument('--seed',default=1004,type=int)
    parser.add_argument('--hidden_layer',default=2,type=int)
    parser.add_argument('--optimizer', default=optimizers.RMSprop(lr=0.001))
    parser.add_argument('--neuron', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--drop_out', default=0, type=int)
    parser.add_argument('--loss', default=losses.categorical_crossentropy)
    parser.add_argument('--activation', default='relu', type=str)

    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
