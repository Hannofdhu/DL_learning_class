"""
尝试使用 2 个或 3 个隐藏层；
尝试使用更多或更少的隐藏单元，比如 32 个、 64 个等；
尝试调整训练轮次 epochs, 批次大小 batch_size
尝试各层采用一定的 Dropout 失活比例；
尝试使用 mse 损失函数代替 categorical_crossentropy
尝试使用 tanh 激活（这种激活在神经网络早期非常流行）代替 relu
尝试使用优化器 Adam 替代 rmsprop

根据上述，共有2的七次方种尝试结果
"""
from tensorflow import  keras
import argparse
import random
#函数参数mode
def main(args):
    #build model
    model = keras.models.Sequential()
    if args.hidden_layer==2:
        model.add(keras.layers.Dense(args.neuron,activation=args.activation,input_shape=(28*28,)))
        model.add(keras.layers.Dense(args.neuron, activation=args.activation))
        model.add(keras.layers.Dropout(args.drop_out))
        model.add(keras.layers.Dense(10,activation='softmax'))
    elif args.hidden_layer==3:
        model.add(keras.layers.Dense(args.neuron, activation=args.activation,input_shape=(28*28,)))
        model.add(keras.layers.Dense(args.neuron, activation=args.activation))
        model.add(keras.layers.Dropout(args.drop_out))
        model.add(keras.layers.Dense(args.neuron, activation=args.activation))
        model.add(keras.layers.Dropout(args.drop_out))
        model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss=args.loss,optimizer=args.optimizer,metrics=['accuracy'])

    #images_processing
    (train_images,train_labels),(test_images,test_labels) = keras.datasets.mnist.load_data()
        #train_images是（60000，28，28)的3D张量，而全连接层只接受1D张量
    train_images = train_images.reshape((60000,28*28))
    test_images = test_images.reshape((10000,28 * 28))
        #极大极小值归一化
    train_images = train_images.astype('float32')/255
    test_images = test_images.astype('float32')/255
        #one-hot编码
    train_labels_one = keras.utils.to_categorical(train_labels)
    test_labels_one = keras.utils.to_categorical(test_labels)

    #model_fit
    model.fit(train_images,train_labels_one,epochs=args.epochs,\
          batch_size=args.batch_size,verbose=2,validation_data=(test_images,test_labels_one))

    #model_evaluate
    loss,accuracy = model.evaluate(test_images,test_labels_one,verbose=2)
    print(f'loss={loss},accuracy={accuracy}')

    #model_prediction
    classes = model.predict(test_images,batch_size=args.batch_size,verbose=2)
    print('测试数：',len(classes))
    print('分类概率:',classes)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #config
    parser.add_argument('--seed',default=1004,type=int)
    parser.add_argument('--hidden_layer',default=2,type=int)
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--neuron', default=64, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--drop_out', default=0.25, type=int)
    parser.add_argument('--loss', default='categorical_crossentropy', type=str)
    parser.add_argument('--activation', default='relu', type=str)

    args = parser.parse_args()
    random.seed(args.seed)
    main(args)

#均为默认值时：loss=0.13496506214141846,accuracy=0.9638000130653381
#三个隐藏层：loss=0.29224058985710144,accuracy=0.9186000227928162
#10个epoch：loss=0.1204235851764679,accuracy=0.9689000248908997
#mse(均方误差)损失函数：loss=0.013858809135854244,accuracy=0.9258000254631042
#tanh激活：loss=0.09488875418901443,accuracy=0.9725000262260437
#adam优化器：loss=0.1633801907300949,accuracy=0.954200029373169

