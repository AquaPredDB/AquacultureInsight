
'''
欢迎关注《淘个代码》公众号。
如代码有问题，请公众号后台留言问题！不要问在吗在吗。
直接截图留言问题！
'''
# In[1]:
###
'''
注意！SVR不可以多步预测！这是SVR定义决定的！！
'''
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
from sklearn.svm import SVR
import warnings
from prettytable import PrettyTable #可以优美的打印表格结果
warnings.filterwarnings("ignore") #取消警告


dataset=pd.read_csv("共享单车租赁数据集.csv",encoding='gb2312')
# 使用pandas模块的read_csv函数读取名为"共享单车租赁数据集.csv"的文件。
# 参数'encoding'设置为'gbk'，这通常用于读取中文字符，确保文件中的中文字符能够正确读取。
# 读取的数据被存储在名为'dataset'的DataFrame变量中。
print(dataset)#显示dataset数据

values = dataset.values[:,2:] #只取第2列数据，要写成1:2；只取第3列数据，要写成2:3，取第2列之后(包含第二列)的所有数据，写成 1：

# 把数据集分为训练集和测试集
values = np.array(values)
# 将前面处理好的DataFrame（data）转换成numpy数组，方便后续的数据操作。
num_samples = values.shape[0]
per = np.random.permutation(num_samples)		#打乱后的行号

n_train_number = per[:int(num_samples * 0.8)]  #选择80%作为训练集
n_test_number = per[int(num_samples * 0.8):]    #选择80%作为训练集
# 计算训练集的大小。
# 设置70%作为训练集
# int(...) 确保得到的训练集大小是一个整数。
# 先划分数据集，在进行归一化，这才是正确的做法！
Xtrain = values[n_train_number, :-1]  #取特征列
Ytrain = values[n_train_number, -1]  #取最后一列为目标列
Ytrain = Ytrain.reshape(-1,1)

Xtest = values[n_test_number, :-1]
Ytest = values[n_test_number,  -1]
Ytest = Ytest.reshape(-1,1)

# 对训练集和测试集进行归一化
m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)  # 注意fit_transform() 和 transform()的区别
vp_test = m_in.transform(Xtest)  # 注意fit_transform() 和 transform()的区别

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)  # 注意fit_transform() 和 transform()的区别
vt_test = m_out.transform(Ytest)  # 注意fit_transform() 和 transform()的区别




model = SVR(C=30, epsilon=0.05, gamma='auto')
model.fit(vp_train, vt_train)

# 使用训练数据（vp_train和vt_train）来训练模型。
# train_X是输入特征，vt_train是目标变量。
# fit方法将会应用前面定义的网络结构和训练参数来训练模型。



# In[14]:
# 作出预测
yhat = model.predict(vp_test)
# 使用模型对测试集的输入特征(vp_test)进行预测。
# yhat是模型预测的输出值。
yhat = yhat.reshape(-1, 1)
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



mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(Ytest, predicted_data, 1)
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


plt.rcParams['axes.unicode_minus'] = False
# 设置matplotlib的配置，用来正常显示负号。
# 使用赛博朋克风样式
# plt.style.use('cyberpunk')
# 创建一个图形对象，并设置大小为10x2英寸，分辨率为300dpi。
plt.figure(figsize=(8, 2), dpi=300)
x = range(1, len(predicted_data) + 1)
# # 创建x轴的值，从1到实际值列表的长度。
# plt.xticks(x[::int((len(predicted_data)+1))])
# 设置x轴的刻度，每几个点显示一个刻度。
plt.tick_params(labelsize=5)  # 改变刻度字体大小
# 设置刻度标签的字体大小。
plt.plot(x, predicted_data, linestyle="--",linewidth=0.8, label='predict',marker = "o",markersize=2)
# 绘制预测值的折线图，线型为虚线，线宽为0.5，标签为'predict'。

plt.plot(x, Ytest, linestyle="-", linewidth=0.5,label='Real',marker = "x",markersize=2)
# 绘制实际值的折线图，线型为直线，线宽为0.5，标签为'Real'。

plt.rcParams.update({'font.size': 5})  # 改变图例里面的字体大小
# 更新图例的字体大小。

plt.legend(loc='upper right', frameon=False)
# 显示图例，位置在图形的右上角，没有边框。

plt.xlabel("Sample points", fontsize=5)
# 设置x轴标签为"样本点"，字体大小为5。

plt.ylabel("value", fontsize=5)
# 设置y轴标签为"值"，字体大小为5。

plt.title(f"The prediction result of bagging :\nMAPE: {mape(Ytest, predicted_data)} %",fontsize=5)


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


