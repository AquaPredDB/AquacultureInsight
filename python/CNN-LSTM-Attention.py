
'''
欢迎关注《淘个代码》公众号。
如代码有问题，请公众号后台留言问题！不要问在吗在吗。
直接截图留言问题！
'''
# In[1]:

# 调用相关库
import os  # 导入os模块，用于操作系统功能，比如环境变量
import math  # 导入math模块，提供基本的数学功能
import pandas as pd  # 导入pandas模块，用于数据处理和分析
import openpyxl
from math import sqrt  # 从math模块导入sqrt函数，用于计算平方根
from numpy import concatenate  # 从numpy模块导入concatenate函数，用于数组拼接
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy模块，用于数值计算
# import tensorflow as tf  # 导入tensorflow模块，用于深度学习
from sklearn.preprocessing import MinMaxScaler  # 导入sklearn中的MinMaxScaler，用于特征缩放
from sklearn.preprocessing import StandardScaler  # 导入sklearn中的StandardScaler，用于特征标准化
from sklearn.preprocessing import LabelEncoder  # 导入sklearn中的LabelEncoder，用于标签编码
from sklearn.metrics import mean_squared_error  # 导入sklearn中的mean_squared_error，用于计算均方误差
from tensorflow.keras.layers import *  # 从tensorflow.keras.layers导入所有层，用于构建神经网络
from tensorflow.keras.models import *  # 从tensorflow.keras.models导入所有模型，用于构建和管理模型
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score  # 导入额外的评估指标
from pandas import DataFrame  # 从pandas导入DataFrame，用于创建和操作数据表
from pandas import concat  # 从pandas导入concat函数，用于DataFrame的拼接
import keras.backend as K  # 导入keras的后端接口
from scipy.io import savemat, loadmat  # 从scipy.io导入savemat和loadmat，用于MATLAB文件的读写
from sklearn.neural_network import MLPRegressor  # 从sklearn.neural_network导入MLPRegressor，用于创建多层感知器回归模型
from keras.callbacks import LearningRateScheduler  # 从keras.callbacks导入LearningRateScheduler，用于调整学习率
from tensorflow.keras import Input, Model, Sequential  # 从tensorflow.keras导入Input, Model和Sequential，用于模型构建
import mplcyberpunk
from qbstyles import mpl_style
import warnings
from prettytable import PrettyTable #可以优美的打印表格结果
warnings.filterwarnings("ignore") #取消警告


dataset=pd.read_csv("电力负荷预测数据1.csv",encoding='gb2312')
# 使用pandas模块的read_csv函数读取名为"农林牧渔.csv"的文件。
# 参数'encoding'设置为'gbk'，这通常用于读取中文字符，确保文件中的中文字符能够正确读取。
# 读取的数据被存储在名为'dataset'的DataFrame变量中。
print(dataset)#显示dataset数据


# 单输入单步预测，就让values等于某一列数据，n_out = 1，n_in, num_samples, scroll_window 根据自己情况来
# 单输入多步预测，就让values等于某一列数据，n_out > 1，n_in, num_samples, scroll_window 根据自己情况来
# 多输入单步预测，就让values等于多列数据，n_out = 1，n_in, num_samples, scroll_window 根据自己情况来
# 多输入多步预测，就让values等于多列数据，n_out > 1，n_in, num_samples, scroll_window 根据自己情况来
values = dataset.values[:,1:] #只取第2列数据，要写成1:2；只取第3列数据，要写成2:3，取第2列之后(包含第二列)的所有数据，写成 1：
# 从dataset DataFrame中提取数据。
# dataset.values将DataFrame转换为numpy数组。
# [:,1:]表示选择所有行（:）和从第二列到最后一列（1:）的数据。
# 这样做通常是为了去除第一列，这在第一列是索引或不需要的数据时很常见。



# 确保所有数据是浮动的
values = values.astype('float32')
# 将values数组中的数据类型转换为float32。
# 这通常用于确保数据类型的一致性，特别是在准备输入到神经网络模型中时。




def data_collation(data, n_in, n_out, or_dim, scroll_window, num_samples):
    res = np.zeros((num_samples,n_in*or_dim+n_out))
    for i in range(0, num_samples):
        h1 = values[scroll_window*i: n_in+scroll_window*i,0:or_dim]
        h2 = h1.reshape( 1, n_in*or_dim)
        h3 = values[n_in+scroll_window*(i) : n_in+scroll_window*(i)+n_out,-1].T
        h4 = h3[np.newaxis, :]
        h5 = np.hstack((h2,h4))
        res[i,:] = h5
    return res
# 关于此函数怎么用，下面详细举例介绍：
# 构造数据，这个函数可以实现单输入单输出，单输入多输出，多输入单输出，和多输入多输出。
# 举个例子：
# 假如原始数据为,其中务必使得数据前n-1列都为特征，最后一列为输出
# [0.74	0.8	0.23 750.75
# 0.74 0.87 0.15 716.94
# 0.74 0.87 0.15 712.77
# 0.74 0.8 0.15 684.86
# 0.74 0.8 0.15 728.79
# 0.72 0.87 0.08 742.81
# 0.71 0.99 0.16 751.3]

#（多输入多输出为例），假如n_in = 2，n_out=2，scroll_window=1
# 输入前2行数据的特征，预测未来2个时刻的数据，滑动窗口为1。
# 使用此函数后，数据会变成：
# 【0.74 0.8 0.23 750.75  0.74	0.87 0.15 716.94 712.77 684.86
# 0.74 0.87 0.15 716.94 0.74 0.87	0.15 712.77  684.86 728.79
# 0.74 0.87 0.15 712.77 0.74 0.8 0.15 684.86 728.79 742.81】

# 假如n_in = 2，n_out=1，scroll_window=2
# 输入前2行数据的特征，预测未来1个时刻的数据，滑动窗口为2。
# 使用此函数后，数据会变成：
# 【0.74 0.8 0.23 750.75  0.74	0.87 0.15 716.94 712.77
# 0.74 0.87	0.15 712.77  0.74 0.8 0.15 684.86 728.79
# 0.74 0.8 0.15 728.79 0.72	0.87 0.08 742.81 751.3】
#写到这里相比大家已经完全明白次函数的用法啦！欢迎关注《淘个代码》公众号！获取更多代码！
#单输入单输出，和单输入多输出也是这么个用法！单输入无非就是数据维度变低了而已。欢迎关注《淘个代码》公众号！获取更多代码！



# In[7]:
# 这里来个多特征输入，单步预测的案例
n_in = 5  # 输入前5行的数据
n_out = 2  # 预测未来2步的数据
or_dim = values.shape[1]        # 记录特征数据维度
num_samples = 2000  # 可以设定从数据中取出多少个点用于本次网络的训练与测试。
scroll_window = 1  #如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
res = data_collation(values, n_in, n_out, or_dim, scroll_window, num_samples)
# 把数据集分为训练集和测试集
values = np.array(res)
# 将前面处理好的DataFrame（data）转换成numpy数组，方便后续的数据操作。

n_train_number = int(num_samples * 0.85)
# 计算训练集的大小。
# 设置80%作为训练集
# int(...) 确保得到的训练集大小是一个整数。
# 先划分数据集，在进行归一化，这才是正确的做法！
Xtrain = values[:n_train_number, :n_in*or_dim]
Ytrain = values[:n_train_number, n_in*or_dim:]


Xtest = values[n_train_number:, :n_in*or_dim]
Ytest = values[n_train_number:,  n_in*or_dim:]

# 对训练集和测试集进行归一化
m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)  # 注意fit_transform() 和 transform()的区别
vp_test = m_in.transform(Xtest)  # 注意fit_transform() 和 transform()的区别

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)  # 注意fit_transform() 和 transform()的区别
vt_test = m_out.transform(Ytest)  # 注意fit_transform() 和 transform()的区别



# In[10]:

vp_train = vp_train.reshape((vp_train.shape[0], n_in, or_dim))
# 将训练集的输入数据vp_train重塑成三维格式。
# 结果是一个三维数组，其形状为[样本数量, 时间步长, 特征数量]。

vp_test = vp_test.reshape((vp_test.shape[0], n_in, or_dim))
# 将训练集的输入数据vp_test重塑成三维格式。
# 结果是一个三维数组，其形状为[样本数量, 时间步长, 特征数量]。


# In[11]:


from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional, LayerNormalization, Input, Conv1D, MaxPooling1D, Reshape
# 从keras.layers模块导入多种层类型。
# Dense是用于创建全连接层的类。
# Activation是用于添加激活函数的层。
# Dropout是用于减少过拟合的丢弃层。
# LSTM是长短时记忆网络层，用于处理序列数据。
# Bidirectional是用于创建双向LSTM层的包装器。
# LayerNormalization是用于层级归一化的类。
# Input是用于模型输入层的函数。

from tensorflow.keras.models import Model
# 从tensorflow.keras.models模块导入Model类。
# Model是用于创建Keras函数式API模型的类。

from sklearn.model_selection import KFold
# 从sklearn.model_selection模块导入KFold类。
# KFold是一种交叉验证方法，用于评估模型的泛化能力。


# In[12]:


def attention_layer(inputs, time_steps):
    # 自定义的注意力机制层
    a = Permute((2, 1))(inputs)  # 将特征维度和时间步维度互换
    a = Dense(time_steps, activation='softmax')(a)  # 使用Dense层计算时间步的权重
    a_probs = Permute((2, 1), name='attention_vec')(a)  # 再次互换维度，使其与原始输入对齐
    output_attention_mul = Multiply()([inputs, a_probs])  # 将注意力权重应用到输入上
    return output_attention_mul

def cnn_lstm_attention_model():
    # 定义一个包含CNN, LSTM和注意力机制的模型
    inputs = Input(shape=(vp_train.shape[1], vp_train.shape[2]))
    conv1d = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
    maxpooling = MaxPooling1D(pool_size=2)(conv1d)
    reshaped = Reshape((-1, 64 * maxpooling.shape[1]))(maxpooling)  # 重塑形状以适应LSTM层
    lstm_out = LSTM(128, return_sequences=True)(reshaped)  # LSTM层
    attention_out = attention_layer(lstm_out, time_steps=reshaped.shape[1])  # 注意力机制层
    attention_flatten = Flatten()(attention_out)  # 展平输出以适应全连接层
    outputs = Dense(vt_train.shape[1])(attention_flatten)  # 全连接层

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='Adam')
    model.summary()

    return model

model = cnn_lstm_attention_model()


history = model.fit(vp_train, vt_train, batch_size=32, epochs=50, validation_split=0.25, verbose=2)
# 训练模型。指定批处理大小为32，训练轮数为50，将25%的数据用作验证集。
# verbose=2表示在训练过程中会输出详细信息。


# In[13]:


# 绘制历史数据
plt.plot(history.history['loss'], label='train')
# 绘制训练过程中的损失曲线。
# history.history['loss']获取训练集上每个epoch的损失值。
# 'label='train''设置该曲线的标签为'train'。

plt.plot(history.history['val_loss'], label='test')
# 绘制验证过程中的损失曲线。
# history.history['val_loss']获取验证集上每个epoch的损失值。
# 'label='test''设置该曲线的标签为'test'。
plt.legend()
# 显示图例，方便识别每条曲线代表的数据集。
plt.show()
# 展示绘制的图像。


# In[14]:
# 作出预测
yhat = model.predict(vp_test)
# 使用模型对测试集的输入特征(vp_test)进行预测。
# yhat是模型预测的输出值。
yhat = yhat.reshape(num_samples-n_train_number, n_out)
# 将预测值yhat重塑为二维数组，以便进行后续操作。

predicted_data = m_out.inverse_transform(yhat)  # 反归一化


def mape(y_true, y_pred):
    # 定义一个计算平均绝对百分比误差（MAPE）的函数。
    record = []
    for index in range(len(y_true)):
        # 遍历实际值和预测值。
        temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
        # 计算单个预测的MAPE。
        record.append(temp_mape)
        # 将MAPE添加到记录列表中。
    return np.mean(record) * 100
    # 返回所有记录的平均值，乘以100得到百分比。

def evaluate_forecasts(Ytest, predicted_data, n_out):
    # 定义一个函数来评估预测的性能。
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    # 初始化存储各个评估指标的字典。
    table = PrettyTable(['测试集指标','MSE', 'RMSE', 'MAE', 'MAPE','R2'])
    for i in range(n_out):
        # 遍历每一个预测步长。每一列代表一步预测，现在是在求每步预测的指标
        actual = [float(row[i]) for row in Ytest]  #一列列提取
        # 从测试集中提取实际值。
        predicted = [float(row[i]) for row in predicted_data]
        # 从预测结果中提取预测值。
        mse = mean_squared_error(actual, predicted)
        # 计算均方误差（MSE）。
        mse_dic.append(mse)
        rmse = sqrt(mean_squared_error(actual, predicted))
        # 计算均方根误差（RMSE）。
        rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted)
        # 计算平均绝对误差（MAE）。
        mae_dic.append(mae)
        MApe = mape(actual, predicted)
        # 计算平均绝对百分比误差（MAPE）。
        mape_dic.append(MApe)
        r2 = r2_score(actual, predicted)
        # 计算R平方值（R2）。
        r2_dic.append(r2)
        if n_out == 1:
            strr = '预测结果指标：'
        else:
            strr = '第'+ str(i + 1)+'步预测结果指标：'
        table.add_row([strr, mse, rmse, mae, str(MApe)+'%', str(r2*100)+'%'])

    return mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table
    # 返回包含所有评估指标的字典。



mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(Ytest, predicted_data, n_out)
# 调用evaluate_forecasts函数。
# 传递实际值(inv_y)、预测值(inv_yhat)以及预测的步数(n_out)作为参数。
# 此函数将计算每个预测步长的RMSE、MAE、MAPE和R2值。

# In[16]:


print(table)#显示预测指标数值


# In[16]:


#%%
## 画结果图
from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 10,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Times New Roman'],#Times New Roman
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

plt.ion()
for ii in range(n_out):

    plt.rcParams['axes.unicode_minus'] = False
    # 设置matplotlib的配置，用来正常显示负号。
    # 使用赛博朋克风样式
    plt.style.use('cyberpunk')
    # 创建一个图形对象，并设置大小为10x2英寸，分辨率为300dpi。
    plt.figure(figsize=(10, 2), dpi=300)
    x = range(1, len(predicted_data) + 1)
    # 创建x轴的值，从1到实际值列表的长度。
    plt.xticks(x[::int((len(predicted_data)+1))])
    # 设置x轴的刻度，每几个点显示一个刻度。
    plt.tick_params(labelsize=5)  # 改变刻度字体大小
    # 设置刻度标签的字体大小。
    plt.plot(x, predicted_data[:,ii], linestyle="--",linewidth=0.5, label='predict')
    # 绘制预测值的折线图，线型为虚线，线宽为0.5，标签为'predict'。

    plt.plot(x, Ytest[:,ii], linestyle="-", linewidth=0.5,label='Real')
    # 绘制实际值的折线图，线型为直线，线宽为0.5，标签为'Real'。

    plt.rcParams.update({'font.size': 5})  # 改变图例里面的字体大小
    # 更新图例的字体大小。

    plt.legend(loc='upper right', frameon=False)
    # 显示图例，位置在图形的右上角，没有边框。

    plt.xlabel("Sample points", fontsize=5)
    # 设置x轴标签为"样本点"，字体大小为5。

    plt.ylabel("value", fontsize=5)
    # 设置y轴标签为"值"，字体大小为5。
    if n_out == 1:     #如果是单步预测
        plt.title(f"The prediction result of CNN-LSTM-Attention :\nMAPE: {mape(Ytest[:, ii], predicted_data[:, ii])} %")
    else:
        plt.title(f"{ii+1} step of CNN-LSTM-Attention prediction\nMAPE: {mape(Ytest[:,ii], predicted_data[:,ii])} %")
    # plt.xlim(xmin=600, xmax=700)  # 显示600-1000的值   局部放大有利于观察
    # 如果需要，可以取消注释这行代码，以局部放大显示600到700之间的值。

    # plt.savefig('figure/预测结果图.png')
    # 如果需要，可以取消注释这行代码，以将图形保存为PNG文件。

plt.ioff()  # 关闭交互模式
plt.show()
# 显示图形。

'''
欢迎关注《淘个代码》公众号。
如代码有问题，请公众号后台留言问题！不要问在吗在吗。
直接截图留言问题！
'''


