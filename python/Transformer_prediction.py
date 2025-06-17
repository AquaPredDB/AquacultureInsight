import warnings
import pandas as pd  # 导入pandas模块，用于数据处理和分析
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # 导入sklearn中的MinMaxScaler，用于特征缩放
from prettytable import PrettyTable #可以优美的打印表格结果
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score  # 导入额外的评估指标
from math import sqrt  # 从math模块导入sqrt函数，用于计算平方根
import argparse
import time
import torch
import torch.nn as nn
warnings.filterwarnings("ignore") #取消警告


'''
    ## ☆☆☆☆注意 此程序与其他程序不同，此程序是基于Pytorch构建的各大网络！☆☆☆☆
'''


## 设置每个网络必要的参数，这里的parser一般不需要变动。
parser = argparse.ArgumentParser()
parser.add_argument('--vision', type=bool, default=True)
parser.add_argument('--train_test_ratio', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=34)
# model
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--n_layers', type=int, default=2)

##transformer
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--hidden_space', type=int, default=32)
# training
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--seed', type=int, default=1)
# optimizer
parser.add_argument('--lr', type=float, default=5e-4, help='Adam learning rate')
args = parser.parse_args(args=[])
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset=pd.read_excel("风电场功率预测.xlsx")
# dataset=pd.read_csv("电力负荷预测数据2.csv",encoding='gb2312')
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
# 这里来个多特征输入，多步预测的案例
n_in = 5  # 输入前5行的数据
n_out = 1  # 预测未来1步的数据
or_dim = values.shape[1]        # 记录特征数据维度
num_samples = 3000  # 可以设定从数据中取出多少个点用于本次网络的训练与测试。
scroll_window = 1  #如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
res = data_collation(values, n_in, n_out, or_dim, scroll_window, num_samples)
# 把数据集分为训练集和测试集
values = np.array(res)
# 将前面处理好的DataFrame（data）转换成numpy数组，方便后续的数据操作。

n_train_number = int(num_samples * 0.9)
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

vp_train = vp_train.reshape((vp_train.shape[0], n_in,or_dim))
# 将训练集的输入数据vp_train重塑成三维格式。
# 结果是一个三维数组，其形状为[样本数量, 时间步长, 特征数量]。

vp_test = vp_test.reshape((vp_test.shape[0],n_in, or_dim))
# 将训练集的输入数据vp_test重塑成三维格式。
# 结果是一个三维数组，其形状为[样本数量, 时间步长, 特征数量]。


# 设置超参数
input_dim = or_dim  #输入维度
output_dim = n_out  #输出维度
hidden_dim = 128   #Feed Forward层（Attention后面的全连接网络）的隐藏层的神经元数量。该值越大，网络参数量越多，计算量越大。
num_layers = 2  #层数

# 转换为torch数据

X_TRAIN = torch.from_numpy(vp_train).type(torch.Tensor)
Y_TRAIN = torch.from_numpy(vt_train).type(torch.Tensor)
X_TEST = torch.from_numpy(vp_test).type(torch.Tensor)
Y_TEST = torch.from_numpy(vt_test).type(torch.Tensor)



class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs,hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space=hidden_space

        # Transformer 的 Encoder 部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,  # 输入特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # 将 Encoder 的输出通过一个全连接层转换为所需的输出维度
        self.output_layer = nn.Linear(hidden_space, num_outputs)
        self.transform_layer=nn.Linear(input_dim, hidden_space)

    def forward(self, x):
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)

        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)
        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 取最后一个时间步的输出
        x = x[-1, :, :]

        # 全连接层生成最终输出
        x = self.output_layer(x)
        return x




model = TimeSeriesTransformer(input_dim=input_dim, num_heads=args.num_heads, num_layers=num_layers,
                              num_outputs=output_dim, hidden_space=args.hidden_space, dropout_rate=args.dropout)

## 损失设定
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

## 统计MSE均方误差和R2决定系数
MSE_hist = np.zeros(args.num_epochs)
R2_hist = np.zeros(args.num_epochs)

##开始时间统计和结果保存
start_time = time.time()
result = []

##循环训练
for t in range(args.num_epochs):
    y_train_pred = model(X_TRAIN)
    ##损失计算
    loss = criterion(y_train_pred, Y_TRAIN)
    ##R2计算
    R2 = r2_score(y_train_pred.detach().numpy(), Y_TRAIN.detach().numpy())
    print("Epoch ", t, "MSE: ", loss.item(), 'R2', R2.item())

    # 统计每个时间步的损失和R2指标
    MSE_hist[t] = loss.item()
    if R2 < 0:
        R2 = 0
    R2_hist[t] = R2
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time() - start_time
print("Training time: {}".format(training_time))



# 测试集测试
y_test_pred = model(X_TEST)


yhat = y_test_pred.detach().numpy()


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
    return np.mean(record)
    # 返回所有记录的平均值。

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
    # 创建一个图形对象，并设置大小为10x2英寸，分辨率为300dpi。
    plt.figure(figsize=(10, 2), dpi=300)
    x = range(1, len(predicted_data) + 1)
    # 创建x轴的值，从1到实际值列表的长度。
    # plt.xticks(x[::int((len(predicted_data)+1))])
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
        plt.title(f"The prediction result of Transformer :\nMAPE: {mape(Ytest[:, ii], predicted_data[:, ii])} %")
    else:
        plt.title(f"{ii+1} step of Transformer prediction\nMAPE: {mape(Ytest[:,ii], predicted_data[:,ii])} %")
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


