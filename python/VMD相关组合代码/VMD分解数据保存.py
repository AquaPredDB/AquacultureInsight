

# In[1]:


import matplotlib.pyplot as plt
# 导入matplotlib的pyplot模块，用于数据可视化。
import tensorflow as tf
# 导入tensorflow库，用于深度学习模型的构建和训练
from vmdpy import VMD 
# 从vmdpy库导入VMD，用于变分模态分解
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


def vmd_decompose(series, num_modes, alpha, tau, DC, init, tol):
    # 定义变分模态分解（VMD）函数，具有多个参数。
    # series：待分解的时间序列。
    # num_modes：分解的模态数量。
    # alpha, tau, K, DC, init, tol：VMD算法的参数。

    decom = VMD(series.values,K =num_modes, alpha=alpha, tau=tau, DC=DC, init=init, tol=tol)
    # 使用VMD方法对输入的时间序列进行分解。

    vmd_result, *_ = decom
    # 从VMD返回的结果中获取分解后的模态。

    df_vmd = pd.DataFrame(vmd_result.T)
    # 将分解后的模态转换为DataFrame。

    df_vmd.columns = ['imf' + str(i+1) for i in range(len(df_vmd.columns))]
    # 为DataFrame的每一列命名，表示每个模态。

    return df_vmd
    # 返回分解结果。

# 调用函数并传入参数
df_vmd_result = vmd_decompose(series=series_close, num_modes=8, alpha=2000, tau=0, DC=1, init=1, tol=1e-7)
# 使用vmd_decompose函数对series_close进行VMD分解。

# 可视化VMD分解结果
fig, axs = plt.subplots(nrows=len(df_vmd_result.columns), figsize=(10, 6), sharex=True)
# 创建一个绘图对象和多个子图对象。

for i, col in enumerate(df_vmd_result.columns):
    axs[i].plot(df_vmd_result[col])
    axs[i].set_title(col)
    # 遍历每个模态并绘制在子图上。

plt.suptitle('VMD Decomposition')
# 设置图表的总标题。

plt.xlabel('Time')
# 设置x轴的标签。

plt.show()
# 显示图表。


# In[7]:


print(df_vmd_result)


# In[8]:


df_vmd_result.to_excel("VMD.xlsx",index=False)#保存数据为VMD.xlsx


# In[ ]:




