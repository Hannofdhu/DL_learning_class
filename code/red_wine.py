#导包
import pandas as pd
import numpy as np
from tensorflow.keras.utils import  to_categorical
from tensorflow.keras import optimizers,losses,metrics,callbacks,layers,models
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import copy
#数据读取
wine_data_red = pd.read_csv(
    '/Users/hann/Documents/ju_code/DL_learning_class/data/葡萄酒数据集/winequality-red.csv', sep=';')
wine_data_white = pd.read_csv(
    '/Users/hann/Documents/ju_code/DL_learning_class/data/葡萄酒数据集/winequality-red.csv', sep=';')

winequality_red_data =wine_data_red.iloc[:,:11]
winequality_red_label =wine_data_red.iloc[:,11]

winequality_white_data =wine_data_white.iloc[:,:11]
winequality_white_label = wine_data_white.iloc[:,11]

#异常值检测：定义去除异常值函数,daset表示数据集，保留num个标准差内的数据,输出时需要删除数据的索引值
def remove_outliers(dataset,num) :
    up_lim=dataset.mean()+num*dataset.std()
    low_lim=dataset.mean()-num*dataset.std()
    drop_list=[]
    for i in range(len(dataset)):
        if 'True' in str(dataset[i:i+1] < low_lim) or 'True' in str(dataset[i:i+1] > up_lim):
            drop_list.append(i)
    return drop_list

red_data_washed=winequality_red_data.drop(index=remove_outliers(winequality_red_data,3))
red_label_washed=winequality_red_label.drop(index=remove_outliers(winequality_red_data,3))

white_data_washed=winequality_white_data.drop(index=remove_outliers(winequality_white_data,3))
white_label_washed=winequality_white_label.drop(index=remove_outliers(winequality_white_data,3))

    #查看清洗后还剩多少样本和标签
#(1536, 11) (1536,) (1536, 11) (1536,)
print(red_data_washed.shape,red_label_washed.shape,white_data_washed.shape, white_label_washed.shape)
#数据标准化
red_data_washed=(red_data_washed-red_data_washed.mean())/red_data_washed.std()
white_data_washed=(white_data_washed-white_data_washed.mean())/white_data_washed.std()
# 将标签值缩放到（0，1）之间
red_label_washed = red_label_washed /10
white_label_washed  = white_label_washed /10
#三集划分
red_train_data, red_val_data, red_test_data = red_data_washed[
    0:900], red_data_washed[901:1200], red_data_washed[1200:]
red_train_label, red_val_label, red_test_label = red_label_washed[
    0:900], red_label_washed[901:1200], red_label_washed[1200:]

white_train_data, white_val_data, white_test_data = white_data_washed[
    0:2700], white_data_washed[2701:3600], white_data_washed[3600:]
white_train_label, white_val_label, white_test_label = white_label_washed[
    0:2700], white_label_washed[2701:3600], white_label_washed[3600:]

#模型架构
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                       input_shape=(11,)))  # 中间层1 32个节点，激活函数relu；输入层为10维数据
    model.add(layers.Dense(64, activation='relu'))  # 中间层2 32个节点，激活函数relu
    # 输出层 1个节点 由于回归分析输出具体数字，不需要激活函数
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse',
              metrics=['mae'])  # 编译模型，优化器:rmsprop，损失函数:mse ，检测指标:mae
    return model
#实例化模型
model=build_model()
his = model.fit(red_train_data, red_train_label, validation_data=(
    red_val_data, red_val_label), batch_size=64, epochs=300, verbose=0)
#曲线平滑
def smooth_curve(points, factor):
    smoothed_points = []
    for point in points:
        if smoothed_points:  # 如果 smoothed_points 非空
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

mae_history=his.history['val_mae']
plt.plot(range(1, len(mae_history) + 1), smooth_curve(mae_history[:],0.9))
plt.show()
#细化显示0～100区间，发现只需训练30个epoch
plt.plot(range(1, 101), smooth_curve(mae_history[0:100],0.95))
plt.show()

model.fit(red_data_washed[0:1200], red_label_washed[0:1200], batch_size=64, epochs=30, verbose=0)
# 评估模型性能 使用测试集评价模型
model.evaluate(red_test_data,red_test_label)
# 查看预测结果
quality = model.predict(red_test_data)
quality_predict = []
for i in range(len(quality)):
    quality_predict.append(float(str(quality[i]).strip('[]')[0:5]))
for i in range(len(quality_predict)):
    quality_predict[i]=round(quality_predict[i]*10)
#print('预测值分别为：', quality_predict,)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————
#导入后端模块
from tensorflow.keras import backend as K
#清理显存
K.clear_session()
#下面是白葡萄酒部分
model=build_model()
his = model.fit(white_train_data, white_train_label, validation_data=(
    white_val_data, white_val_label), batch_size=16, epochs=300, verbose=0)
mae_history=his.history['val_mae']
plt.plot(range(1, len(mae_history) + 1), smooth_curve(mae_history[:],0.95))
plt.show()

#细化显示0～100区间，发现只需训练20个epoch
plt.plot(range(1, 101), smooth_curve(mae_history[0:100],0.95))
plt.show()

model=build_model()
model.fit(white_data_washed[0:2700], white_label_washed[0:2700], batch_size=16, epochs=20, verbose=0)
# 评估模型性能 使用测试集评价模型
model.evaluate(white_test_data,white_test_label)
# 查看预测结果
quality = model.predict(white_test_data)
quality_predict = []
for i in range(len(quality)):
    quality_predict.append(float(str(quality[i]).strip('[]')[0:5]))
for i in range(len(quality_predict)):
    quality_predict[i]=round(quality_predict[i]*10)
#print('预测值分别为：', quality_predict)

