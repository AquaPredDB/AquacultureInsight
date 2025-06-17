#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
# 导入matplotlib的pyplot模块，用于数据可视化。
import tensorflow as tf
# 导入tensorflow库，用于深度学习模型的构建和训练
from PyEMD import EMD, EEMD, CEEMDAN
# 从PyEMD库导入EMD, EEMD, CEEMDAN，用于模态分解
import pandas as pd
# 导入pandas库，用于数据处理和分析
import warnings
# 导入warnings库，用于控制警告消息
warnings.filterwarnings("ignore")
# 设置忽略警告消息，通常用于减少输出中的不必要警告


# In[2]:


# 调用GPU加速
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[3]:


# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False


# In[4]:


df_raw_data = pd.read_csv('电力负荷预测数据2.csv', encoding='gbk')
# 使用pandas的read_csv函数读取CSV文件。
# '股票预测.csv'是文件名。
# usecols=[0,-1]指定只读取CSV文件的第一列和最后一列。
# encoding='gbk'指定文件编码格式为GBK，GBK常用于中文字符编码。
series_close = pd.Series(df_raw_data['power'].values, index=df_raw_data['time'])
# 创建一个pandas的Series对象。
# df_raw_data['power'].values提取'power'列的值作为Series的数据。
# index=df_raw_data['time']设置Series的索引为'time'列的值。


# In[5]:


print(series_close)


# In[6]:


def ceemdan_decompose(series=None, trials=100, num_clusters=3):
    # 定义CEEMDAN分解函数。
    # series: 待分解的时间序列数据。
    # trials: CEEMDAN的试验次数，用于生成噪声。
    # num_clusters: 分解后的模态数量（这个参数在函数内并未使用）。

    decom = CEEMDAN()
    # 创建CEEMDAN对象。

    decom.trials = trials 
    # 设置CEEMDAN对象的试验次数。

    df_ceemdan = pd.DataFrame(decom(series.values).T)
    # 对时间序列数据执行CEEMDAN分解，并将结果转置后转换为DataFrame。

    df_ceemdan.columns = ['imf' + str(i+1) for i in range(len(df_ceemdan.columns))]
    # 为DataFrame的每一列命名，表示每个内在模态函数（IMF）。

    df = pd.DataFrame(df_ceemdan, columns=df_ceemdan.columns)
    # 创建一个新的DataFrame（这一步实际上是多余的，因为df_ceemdan已经是所需的DataFrame了）。

    return df_ceemdan
    # 返回分解结果的DataFrame。


# In[7]:


df_ceemdan = ceemdan_decompose(series_close)
# 对series_close进行CEEMDAN分解，并返回结果的DataFrame。

# 可视化VMD分解结果
fig, axs = plt.subplots(nrows=len(df_ceemdan.columns), figsize=(10, 6), sharex=True)
# 创建一个绘图对象和多个子图对象。

for i, col in enumerate(df_ceemdan.columns):
    axs[i].plot(df_ceemdan[col])
    axs[i].set_title(col)
    # 遍历每个模态并绘制在子图上。

plt.suptitle('CEEMDAN Decomposition')
# 设置图表的总标题。

plt.xlabel('Time')
# 设置x轴的标签。

plt.show()
# 显示图表。


# In[8]:


print(df_ceemdan)


# In[9]:


df_ceemdan.to_excel("CEEMDAN.xlsx",index=False)#保存数据为CEEMDAN.xlsx


# In[ ]:




