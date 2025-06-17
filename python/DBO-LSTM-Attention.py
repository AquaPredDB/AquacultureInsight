
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
import tensorflow as tf  # 导入tensorflow模块，用于深度学习
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
import copy
# 导入copy模块，用于对象的复制。
import random
# 导入random模块，用于生成随机数。
from scipy.io import savemat, loadmat
# 从scipy.io模块导入savemat和loadmat函数，用于读写MATLAB格式的文件。
from numpy import concatenate
# 从numpy模块导入concatenate函数，用于数组的连接。
from matplotlib.pylab import mpl
import warnings
from prettytable import PrettyTable   #可以优美的打印表格结果
warnings.filterwarnings("ignore") #取消警告
# 从matplotlib.pylab模块导入mpl，用于配置matplotlib的一些参数。


dataset=pd.read_csv("电力负荷预测数据1.csv",encoding='gb2312')
# 使用pandas模块的read_csv函数读取名为"农林牧渔.csv"的文件。
# 参数'encoding'设置为'gbk'，这通常用于读取中文字符，确保文件中的中文字符能够正确读取。
# 读取的数据被存储在名为'dataset'的DataFrame变量中。
print(dataset)#显示dataset数据


values = dataset.values[:,1:]
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


from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional, LayerNormalization, Input
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



def attention_layer(inputs, single_attention_vector=False):
    # 注意力机制层的实现
    time_steps = K.int_shape(inputs)[1]  # 获取输入的时间步长
    input_dim = K.int_shape(inputs)[2]  # 获取输入特征的维度
    a = Permute((2, 1))(inputs)  # 将时间步长和特征维度互换，为了后续的处理
    a = Reshape((input_dim, time_steps))(a)  # 重塑形状，以适应Dense层
    a = Dense(time_steps, activation='softmax')(a)  # 使用Dense层和softmax激活函数计算注意力权重
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)  # 如需，减少维度
        a = RepeatVector(input_dim)(a)  # 重复向量以匹配输入的维度
    a_probs = Permute((2, 1), name='attention_vec')(a)  # 再次互换维度，将其变回原来的形状
    output_attention_mul = Multiply()([inputs, a_probs])  # 将注意力权重应用于输入
    return output_attention_mul

def lstm_attention_model():
    # 定义一个包含LSTM和注意力机制的模型
    inputs = Input(shape=(vp_train.shape[1], vp_train.shape[2]))
    # 输入层，定义输入数据的形状
    lstm_out = LSTM(20, return_sequences=True)(inputs)
    # LSTM层，128个神经元，return_sequences=True使得每个时间步的输出都保留
    attention_mul = attention_layer(lstm_out)
    # 应用自定义的注意力层
    attention_flatten = Flatten()(attention_mul)
    # 使用Flatten层将数据展平，以便传递给Dense层
    outputs = Dense(vt_train.shape[1])(attention_flatten)
    # 全连接层，输出维度与vt_train的形状一致

    model = Model(inputs=[inputs], outputs=outputs)
    # 创建模型，定义输入和输出
    model.compile(loss='mse', optimizer='Adam')
    # 编译模型，使用均方误差作为损失函数，优化器为Adam
    model.summary()
    # 输出模型的总结信息

    return model

model = lstm_attention_model()


history = model.fit(vp_train, vt_train, batch_size=32, epochs=20, validation_split=0.25, verbose=2)
# 训练模型。指定批处理大小为16，训练轮数为15，将25%的数据用作验证集。
# verbose=2表示在训练过程中会输出详细信息。


# In[13]:

plt.figure()
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

predicted_data = m_out.inverse_transform(yhat)  # 反归一化




def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


print('MSE:', mean_squared_error(Ytest, predicted_data))

print('RMSE:', np.sqrt(mean_squared_error(Ytest, predicted_data)))

print('MAE:', mean_absolute_error(Ytest, predicted_data))

print('MAPE:', mape(Ytest, predicted_data))

print('R2-score:',r2_score(Ytest, predicted_data))


# In[16]:






'''*****************************以上为标准LSTM-Attention程序*******************************************'''



'''*****************************接下来采用DBO优化LSTM-Attention*****************************************'''

'''
进行适应度计算,以验证集均方差为适应度函数，目的是找到一组超参数 使得网络的误差最小
'''
#  这里优化了学习率和lstm神经元个数两个参数
def fun(pop, P, T, Pt, Tt):
    # 定义适应度函数，pop是一个包含超参数的数组。
    tf.random.set_seed(0)
    # 设置TensorFlow的随机种子以确保实验的可重复性。
    alpha = pop[0]  # 学习率
    # 从pop数组中获取学习率。
    hidden_nodes0 = int(pop[1])  # 第一隐含层神经元，从pop数组中获取第一隐含层神经元的数量。
    #     num_epochs = int(pop[2])#迭代次数
    #     batch_size = int(pop[3])# batchsize
    #     hidden_nodes = int(pop[4])#第二隐含层神经元

    inputs = Input(shape=(vp_train.shape[1], vp_train.shape[2]))
    # 输入层，定义输入数据的形状
    lstm_out = LSTM(hidden_nodes0, return_sequences=True)(inputs)
    # LSTM层，128个神经元，return_sequences=True使得每个时间步的输出都保留

    attention_mul = attention_layer(lstm_out)
    # 应用自定义的注意力层
    attention_flatten = Flatten()(attention_mul)
    # 使用Flatten层将数据展平，以便传递给Dense层
    outputs = Dense(vt_train.shape[1])(attention_flatten)
    # 全连接层，输出维度与vt_train的形状一致
    model = Model(inputs=[inputs], outputs=outputs)
    # 创建模型，定义输入和输出
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    # 编译模型，使用Adam优化器，学习率为alpha，损失函数为均方误差。
    #     model.compile(loss='mse',optimizer='Adam')
    # model.summary()
    # # 输出模型的总结信息
    # model.fit(vp_train, vt_train, batch_size=32, epochs=50, validation_split=0.25, verbose=2)
    model.fit(vp_train, vt_train, epochs=30, batch_size=32, validation_data=(vp_test, vt_test),
              verbose=0, shuffle=False)
    # 训练模型，设置迭代次数为30，批处理大小为32，使用vt_train和vt_test作为验证数据。

    test_pred = model.predict(vp_test)
    # 使用模型对测试数据进行预测。

    MApe_error = mape(vt_test, test_pred)
    # 计算预测值和实际值之间的均方误差。

    return MApe_error,test_pred
    # 返回计算得到的均方误差。


# In[12]:


'''边界检查函数'''
def boundary(pop, lb, ub):
    # 定义一个边界检查函数，确保种群中的个体不超出预定义的边界。
    pop = pop.flatten()
    lb = lb.flatten()
    ub = ub.flatten()
    # 将输入参数扁平化，以便进行元素级操作。

    # 防止跳出范围,除学习率之外 其他的都是整数
    pop = [int(pop[i]) if i > 0 else pop[i] for i in range(lb.shape[0])]
    # 将除了学习率以外的参数转换为整数。

    for i in range(len(lb)):
        if pop[i] > ub[i] or pop[i] < lb[i]:
            # 检查个体是否超出边界。
            if i == 0:
                pop[i] = (ub[i] - lb[i]) * np.random.rand() + lb[i]
                # 如果是学习率，则在边界内随机选择一个值。
            else:
                pop[i] = np.random.randint(lb[i], ub[i])
                # 对于整数参数，随机选择一个边界内的整数值。

    return pop
    # 返回修正后的个体。

''' 种群初始化函数 '''
def initial(pop, dim, ub, lb):
    # 定义一个初始化种群的函数。
    X = np.zeros([pop, dim])
    # 创建一个形状为[种群大小, 维度]的零矩阵。

    for i in range(pop):
        for j in range(dim):
            X[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
            # 在边界内随机初始化每个个体的每个参数。

    return X, lb, ub
    # 返回初始化后的种群及边界。

'''计算适应度函数'''
def CaculateFitness(X, fun, P, T, Pt, Tt):
    # 定义一个计算适应度的函数。
    pop = X.shape[0]
    # 获取种群的大小。
    fitness = np.zeros([pop, 1])
    # 创建一个形状为[种群大小, 1]的零矩阵来存储适应度。

    for i in range(pop):
        fitness[i],pre = fun(X[i, :], P, T, Pt, Tt)
        # 对每个个体调用适应度函数进行计算。

    return fitness
    # 返回计算得到的适应度。

'''适应度排序'''
def SortFitness(Fit):
    # 定义一个对适应度进行排序的函数。
    fitness = np.sort(Fit, axis=0)
    # 按适应度大小进行排序。
    index = np.argsort(Fit, axis=0)
    # 获取排序后的索引。

    return fitness, index
    # 返回排序后的适应度和索引。

'''根据适应度对位置进行排序'''
def SortPosition(X, index):
    # 定义一个根据适应度排序位置的函数。
    Xnew = np.zeros(X.shape)
    # 创建一个与X形状相同的零矩阵。

    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
        # 根据适应度的排序结果重新排列位置。

    return Xnew
    # 返回排序后的位置。


# In[13]:


'''蜣螂优化算法'''
def DBO(P, T, Pt, Tt):
    # 参数设置
    pop = 10  # pop种群数量     #MaxIter和pop这两个参数设置的越大  相对来说寻优出来适应度越好效果越好  ，但是算法运行花的时间就越多
    MaxIter = 20  # MaxIter最大迭代次数
    PballRolling = 0.2  # 滚球蜣螂比例
    PbroodBall = 0.4  # 产卵蜣螂比例
    PSmall = 0.2  # 小蜣螂比例
    Pthief = 0.2  # 偷窃蜣螂比例
    BallRollingNum = int(pop * PballRolling)  # 滚球蜣螂数量
    BroodBallNum = int(pop * PbroodBall)  # 产卵蜣螂数量
    SmallNum = int(pop * PSmall)  # 小蜣螂数量
    ThiefNum = int(pop * Pthief)  # 偷窃蜣螂数量
    dim = 2  # 搜索维度,
    # 优化了两个参数  学习率和神经元个数   Lb和Ub分别为寻优范围上下限
    # 第一个是学习率[0.001 0.01]
    # 第二个是神经元个数[10-100]
    lb = np.array([0.001, 10]).reshape(-1, 1)
    ub = np.array([0.01, 100]).reshape(-1, 1)

    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    # 调用initial函数来初始化种群。
    # 参数pop是种群大小，dim是搜索空间的维度，ub和lb分别是搜索空间的上界和下界。
    # 函数返回初始化的种群X以及更新后的上下界。
    for i in range(pop):
        X[i, :] = boundary(X[i, :], lb, ub)
        # 遍历种群中的每一个个体。
        # 调用boundary函数来确保个体的参数在定义的边界内。
        # boundary函数根据上下界lb和ub对个体X[i, :]的参数进行调整。
    fitness = CaculateFitness(X, fun, P, T, Pt, Tt)
    # 调用CaculateFitness函数来计算种群中每个个体的适应度。
    # 参数X是种群，fun是计算适应度的函数，P, T, Pt, Tt是fun函数所需的额外参数。
    # 函数返回一个数组，包含了种群中每个个体的适应度。
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    # 记录全局最优
    minIndex = np.argmin(fitness)
    # 找到当前适应度最小值的索引，即找到当前最优解的位置。
    GbestScore = copy.copy(fitness[minIndex])
    # 复制当前最优适应度值到GbestScore。
    GbestPositon = np.zeros([1, dim])
    # 初始化全局最优位置矩阵，形状为1行dim列。
    GbestPositon[0, :] = copy.copy(X[minIndex, :])
    # 将当前最优解的位置复制到全局最优位置矩阵。
    Curve = np.zeros([MaxIter, 1])
    # 初始化一个记录每次迭代全局最优适应度值的矩阵。
    result = np.zeros([MaxIter, dim])
    # 初始化一个记录每次迭代全局最优位置的矩阵。
    Xl = copy.deepcopy(X)  # 用于记录X(t-1)
    # 深拷贝当前种群，用于记录上一代种群的位置。

    # 记录当前代种群
    cX = copy.deepcopy(X)
    # 深拷贝当前种群，用于在迭代过程中更新种群的位置。
    cFit = copy.deepcopy(fitness)
    # 深拷贝当前适应度值，用于在迭代过程中更新种群的适应度。

    for t in range(MaxIter):
        print("第" + str(t) + "次迭代")
        # 开始迭代，打印当前迭代次数。

        # 蜣螂滚动
        # 获取种群最差值
        maxIndex = np.argmax(fitness)# 找到当前种群中适应度最差的个体的索引。
        Wort = copy.copy(X[maxIndex, :])# 复制最差个体的位置。
        r2 = np.random.random()# 生成一个随机数，用于后续决策。
        for i in range(0, BallRollingNum):# 遍历所有的滚球蜣螂。
            if r2 < 0.9:# 如果随机数小于0.9，执行一种滚动策略。
                if np.random.random() > 0.5:
                    alpha = 1
                else:
                    alpha = -1
                    # 随机选择一个方向。
                b = 0.3
                k = 0.1# 设置滚动系数b和k。
                X[i, :] = cX[i, :] + b * np.abs(cX[i, :] - Wort) + alpha * k * Xl[i, :]# 根据滚球策略更新位置。
            else:# 如果随机数大于或等于0.9，执行另一种滚动策略。
                theta = np.random.randint(180)# 随机生成一个角度。
                if theta == 0 or theta == 90 or theta == 180:
                    X[i, :] = copy.copy(cX[i, :])# 如果角度为0、90或180度，保持位置不变。
                else:
                    theta = theta * np.pi / 180# 将角度转换为弧度。
                    X[i, :] = cX[i, :] + np.tan(theta) * np.abs(cX[i, :] - Xl[i, :])# 根据另一种滚动策略更新位置。
            X[i, :] = boundary(X[i, :], lb, ub)# 使用边界函数确保位置在可行域内。
            fitness[i],pre = fun(X[i, :],P,T,Pt,Tt)# 重新计算适应度。
            if fitness[i] < GbestScore:# 如果新的适应度比全局最优还好，更新全局最优。
                GbestScore = copy.copy(fitness[i])
                GbestPositon[0, :] = copy.copy(X[i, :])
        # 当前迭代最优
        minIndex = np.argmin(fitness)# 找到当前适应度最小的个体索引。
        GbestB = copy.copy(X[minIndex, :])# 将当前迭代最优位置复制到GbestB。

        # 蜣螂产卵
        R = 1 - t / MaxIter# 计算衰减因子R，随着迭代次数增加而减小
        X1 = GbestB * (1 - R)
        X2 = GbestB * (1 + R)
        # 计算产卵区域的上下界
        Lb = np.zeros(dim)
        Ub = np.zeros(dim)
        # 初始化局部搜索空间的上下界
        for j in range(dim):
            Lb[j] = max(X1[j], lb[j, 0])
            Ub[j] = min(X2[j], ub[j, 0])
            # 计算每个维度的局部搜索空间上下界
        for i in range(BallRollingNum, BallRollingNum + BroodBallNum):
            b1 = np.random.random()
            b2 = np.random.random()
            # 生成两个随机数。
            X[i, :] = GbestB + b1 * (cX[i, :] - Lb) + b2 * (cX[i, :] - Ub)# 根据产卵行为更新位置
            X[i, :] = boundary(X[i, :], lb, ub)# 确保更新后的位置在定义的边界内
            fitness[i],pre = fun(X[i, :],P,T,Pt,Tt)# 重新计算适应度。
            if fitness[i] < GbestScore:
                GbestScore = copy.copy(fitness[i])
                GbestPositon[0, :] = copy.copy(X[i, :])# 如果找到更好的解，更新全局最优。
        # 小蜣螂更新
        R = 1 - t / MaxIter# 重新计算衰减因子R。
        X1 = GbestPositon[0, :] * (1 - R)
        X2 = GbestPositon[0, :] * (1 + R)
        # 计算探索区域的上下界。

        Lb = np.zeros(dim)
        Ub = np.zeros(dim)
        # 重新初始化局部搜索空间的上下界。
        for j in range(dim):
            Lb[j] = max(X1[j], lb[j, 0])
            Ub[j] = min(X2[j], ub[j, 0])
            # 计算每个维度的局部搜索空间上下界。
        for i in range(BallRollingNum + BroodBallNum, BallRollingNum + BroodBallNum + SmallNum):
            C1 = np.random.random([1, dim])
            C2 = np.random.random([1, dim])
            # 生成两个随机数
            X[i, :] = GbestPositon[0, :] + C1 * (cX[i, :] - Lb) + C2 * (cX[i, :] - Ub)# 根据小蜣螂探索行为更新位置
            X[i, :] = boundary(X[i, :], lb, ub)# 确保更新后的位置在定义的边界内。
            fitness[i],pre = fun(X[i, :],P,T,Pt,Tt)# 重新计算适应度。
            if fitness[i] < GbestScore:
                GbestScore = copy.copy(fitness[i])
                GbestPositon[0, :] = copy.copy(X[i, :])# 如果找到更好的解，更新全局最优
        # 当前迭代最优
        minIndex = np.argmin(fitness)# 找到当前适应度最小的个体索引。
        GbestB = copy.copy(X[minIndex, :])# 将当前迭代最优位置复制到GbestB。
        # 偷窃蜣螂更新
        for i in range(pop - ThiefNum, pop):# 遍历所有的偷窃蜣螂。
            g = np.random.randn()# 生成一个标准正态分布的随机数。
            S = 0.5 # 设置偷窃蜣螂的步长系数。
            X[i, :] = GbestPositon[0, :] + g * S * (np.abs(cX[i, :] - GbestB) + np.abs(cX[i, :] - GbestPositon[0, :]))# 根据偷窃蜣螂的行为规则更新位置。
            X[i, :] = boundary(X[i, :], lb, ub)# 确保更新后的位置在定义的边界内。
            fitness[i],pre = fun(X[i, :],P,T,Pt,Tt)# 重新计算适应度。
            if fitness[i] < GbestScore:# 如果找到更好的解，更新全局最优。
                GbestScore = copy.copy(fitness[i])
                GbestPositon[0, :] = copy.copy(X[i, :])
        # 记录t代种群
        Xl = copy.deepcopy(cX)# 保存当前代种群的位置，以便在下一代中使用。
        # 更新当前代种群
        for i in range(pop):# 遍历种群中的每一个个体。
            if fitness[i] < cFit[i]: # 如果个体的新适应度比之前记录的适应度好，则更新。
                cFit[i] = copy.copy(fitness[i])
                cX[i, :] = copy.copy(X[i, :])

        Curve[t] = GbestScore# 记录当前代的全局最优适应度。
        result[t, :] = GbestPositon# 记录当前代的全局最优位置。
        print('第',t,'代寻优最小MAPE是：',GbestScore)
    return GbestPositon, Curve, result# 返回全局最优位置、每代最优适应度值和每代全局最优位置。


# In[14]:


# 开始优化参数
best, trace, result = DBO(vp_train, vt_train, vp_test, vt_test)
# 调用DBO函数进行优化。传入训练和测试数据集，返回最优参数、每次迭代的适应度跟踪和每次迭代的最优结果。

# 保存优化结果
savemat('dbo_LSTM_Attention_para.mat', {'trace': trace, 'best': best, 'result': result})
# 将优化结果保存到MAT文件中。'trace'记录每次迭代的适应度值，'best'是最优参数，'result'是每次迭代的最优结果。

print("LSTM-Attention最优学习率、最佳神经元的参数分别为：", [int(best[i]) if i > 0 else best[i] for i in range(len(best))])
# 打印最优学习率和LSTM层神经元的数量。对于非学习率参数，将其转换为整数。

# 画图
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.figure(figsize=(6, 4), dpi=500)
# 创建一个绘图窗口，设置大小和分辨率。
plt.plot(trace, 'r', linestyle="--", linewidth=0.5)
# 绘制适应度跟踪曲线，颜色为红色，线型为虚线，线宽为0.5。
#plt.xticks(list(range(0, 35, 5)))
# 设置x轴的刻度（如果需要）。
plt.xlabel('迭代次数', fontsize=10)
# 设置x轴标签为“迭代次数”。
plt.ylabel('适应度值', fontsize=10)
# 设置y轴标签为“适应度值”。
plt.show()
# 显示绘制的图形。


'''
# ………………………………………………………………利用蜣螂优化的参数建模……………………………………………………………………………………
'''
tf.random.set_seed(0)# 设置TensorFlow的随机种子，以确保实验的可重复性。
np.random.seed(0)# 设置numpy的随机种子。
pop=loadmat('dbo_LSTM_Attention_para.mat')['best'].reshape(-1,)


ffit,DBO_LA_pred = fun(pop,vp_train, vt_train, vp_test, vt_test)
# 这里的ffit后面用不到
DBO_predicted_data = m_out.inverse_transform(DBO_LA_pred)  # 反归一化

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


plt.rcParams['axes.unicode_minus'] = False
# 设置matplotlib的配置，用来正常显示负号。

# 使用赛博朋克风样式
plt.style.use('cyberpunk')
plt.figure(figsize=(10,2),dpi=300)
# 创建一个图形对象，并设置大小为10x2英寸，分辨率为300dpi。

x = range(1, len(predicted_data) + 1)
# 创建x轴的值，从1到实际值列表的长度。

plt.xticks(x[::int((len(predicted_data)+1)/20)])
# 设置x轴的刻度，每几个点显示一个刻度。

plt.tick_params(labelsize=5)  # 改变刻度字体大小
# 设置刻度标签的字体大小。


plt.plot(x, DBO_predicted_data, linestyle="-.",linewidth=0.5, label='DBO-LSTM-Attention')
# 绘制预测值的折线图，线型为虚线，线宽为0.5，标签为'predict'。

plt.plot(x, predicted_data, linestyle="--",linewidth=0.5, label='LSTM-Attention')
# 绘制预测值的折线图，线型为虚线，线宽为0.5，标签为'predict'。

plt.plot(x, Ytest, linestyle="-", linewidth=0.5,label='Real')
# 绘制实际值的折线图，线型为直线，线宽为0.5，标签为'Real'。

plt.rcParams.update({'font.size': 5})  # 改变图例里面的字体大小
# 更新图例的字体大小。

plt.legend(loc='upper right', frameon=False)
# 显示图例，位置在图形的右上角，没有边框。

plt.xlabel("Sample points", fontsize=5)
# 设置x轴标签为"样本点"，字体大小为5。

plt.ylabel("value", fontsize=5)
# 设置y轴标签为"值"，字体大小为5。

plt.title(f"Prediction results\nLSTM-Attention,MSE: {mape(Ytest, predicted_data)*100} %\nDBO-LSTM-Attention,MSE: {mape(Ytest, DBO_predicted_data)*100} %")
# plt.xlim(xmin=600, xmax=700)  # 显示600-1000的值   局部放大有利于观察
# 如果需要，可以取消注释这行代码，以局部放大显示600到700之间的值。

# plt.savefig('figure/预测结果图.png')
# 如果需要，可以取消注释这行代码，以将图形保存为PNG文件。

plt.show()
# 显示图形。


print('LSTM-Attention,MSE:', mean_squared_error(Ytest, predicted_data))

print('LSTM-Attention,RMSE:', np.sqrt(mean_squared_error(Ytest, predicted_data)))

print('LSTM-Attention,MAE:', mean_absolute_error(Ytest, predicted_data))

print('LSTM-Attention,MAPE:', mape(Ytest, predicted_data))

print('LSTM-Attention,R2-score:',r2_score(Ytest, predicted_data))

print('DBO-LSTM-Attention,MSE:', mean_squared_error(Ytest, DBO_predicted_data))

print('DBO-LSTM-Attention,RMSE:', np.sqrt(mean_squared_error(Ytest, DBO_predicted_data)))

print('DBO-LSTM-Attention,MAE:', mean_absolute_error(Ytest, DBO_predicted_data))

print('DBO-LSTM-Attention,MAPE:', mape(Ytest, DBO_predicted_data))

print('DBO-LSTM-Attention,R2-score:',r2_score(Ytest, DBO_predicted_data))