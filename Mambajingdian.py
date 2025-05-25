import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义选择性状态空间模型 (Selective State Space Model, SSM)
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, seq_len):
        super(SelectiveSSM, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# 定义 Mamba 块
class MambaBlock(nn.Module):
    def __init__(self, d_model, seq_len):
        super(MambaBlock, self).__init__()
        self.ssm = SelectiveSSM(d_model, seq_len)
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)  # 转换为 [batch_size, d_model, seq_len]
        x = self.conv(x)
        x = x.transpose(1, 2)  # 转回 [batch_size, seq_len, d_model]
        x = self.norm(x + residual)
        x = self.activation(x)
        x = self.ssm(x)
        x = self.linear(x)
        return x

# 定义 Mamba 模型
class MambaModel(nn.Module):
    def __init__(self, input_dim, d_model, seq_len, num_layers):
        super(MambaModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.mamba_blocks = nn.ModuleList([MambaBlock(d_model, seq_len) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        x = self.output_layer(x[:, -1, :])  # 取最后一个时间步的输出进行预测
        return x

# 设置参数    Best hyperparameters:  {'d_model': 128, 'num_layers': 2, 'lr': 0.008891429472616062}
file_path = 'data/FC/FC3_I7.csv'
w = 120  # 输入序列长度
length_size = 120  # 预测长度
e = 100  # 迭代次数
batch_size = 256  # 批量大小

# 读取数据
data = pd.read_csv(file_path)
data = data.iloc[:, 1:]  # 去除时间列
data_dim = data.shape[1]
scaler = preprocessing.MinMaxScaler()

# 对所有数据进行缩放
data_scaled = scaler.fit_transform(data)

data_length = len(data_scaled)

# 数据集划分比例
train_size = int(0.8 * data_length)
val_size = int(0.1 * data_length)
test_size = data_length - train_size - val_size

# 划分数据集
data_train = data_scaled[:train_size, :]
data_val = data_scaled[train_size:train_size + val_size, :]
data_test = data_scaled[train_size + val_size:, :]

n_feature = data_dim

def data_loader(w, length_size, batch_size, data):
    seq_len = w
    sequence_length = seq_len + length_size
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)
    x_train = result[:, :-length_size, :]
    y_train = result[:, -length_size:, -1]
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], data_dim))
    y_train = np.reshape(y_train, (y_train.shape[0], -1))

    X_train, y_train = torch.tensor(X_train).to(torch.float32).to(device), torch.tensor(y_train).to(torch.float32).to(device)
    ds = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dataloader, X_train, y_train

dataloader_train, X_train, y_train = data_loader(w, length_size, batch_size, data_train)
dataloader_val, X_val, y_val = data_loader(w, length_size, batch_size, data_val)
dataloader_test, X_test, y_test = data_loader(w, length_size, batch_size, data_test)

# 模型训练
def model_train():
    net = MambaModel(input_dim=n_feature, d_model=128, seq_len=w, num_layers=2).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.00889)

    best_val_loss = float('inf')
    best_model_path = 'checkpoint/best_MambaModel_i7_30.pt'

    for epoch in range(e):
        net.train()
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            preds = net(datapoints)
            labels = labels[:, -1].view(-1, 1)  # 调整标签形状以匹配预测值
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}/{e}, Step: {i}, Training Loss: {loss.item():.6f}")

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for datapoints, labels in dataloader_val:
                preds = net(datapoints)
                labels = labels[:, -1].view(-1, 1)  # 调整标签形状以匹配预测值
                loss = criterion(preds, labels)
                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        print(f"Epoch: {epoch + 1}/{e}, Validation Loss: {val_loss:.6f}")

        # 保存验证损失最小的模型，并包括超参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': net.state_dict(),
                'hyperparameters': {
                    'input_dim': n_feature,
                    'd_model': 128,
                    'seq_len': w,
                    'num_layers': 2
                }
            }, best_model_path)

    print("Training complete. Best validation loss: {:.6f}".format(best_val_loss))
    return net


# 模型测试
def model_test():
    # 加载模型和超参数
    checkpoint = torch.load('checkpoint/best_MambaModel_i7_30.pt')
    if 'hyperparameters' not in checkpoint:
        raise KeyError("'hyperparameters' key not found in checkpoint. Please ensure the model is saved correctly with hyperparameters.")

    # 加载超参数
    hyperparameters = checkpoint['hyperparameters']

    # 使用保存的超参数创建模型
    net = MambaModel(input_dim=hyperparameters['input_dim'],
                     d_model=hyperparameters['d_model'],
                     seq_len=hyperparameters['seq_len'],
                     num_layers=hyperparameters['num_layers']).to(device)

    # 加载模型参数
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    with torch.no_grad():
        X_test_gpu = X_test.to(device)
        pred = net(X_test_gpu)
        pred = pred.detach().cpu().numpy()
        true = y_test.detach().cpu().numpy()

    # 对预测值和真实值进行逆变换
    target_scaler = preprocessing.MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[-1:], scaler.scale_[-1:]

    pred_uninverse = target_scaler.inverse_transform(pred)
    true_uninverse = target_scaler.inverse_transform(true)

    return true_uninverse, pred_uninverse



# 评估
if __name__ == "__main__":
    net = model_train()
    true, pred = model_test()

    # 误差评估
    combined_results = np.column_stack((true[:, 0], pred[:, 0]))
    time = np.arange(len(combined_results))

    plt.figure(figsize=(12, 3))
    plt.plot(time, combined_results[:, 0], c='red', linestyle='-', linewidth=1, label='True DO')
    plt.plot(time, combined_results[:, 1], c='blue', linestyle='--', linewidth=1, label='Predicted DO')
    plt.title('Mamba Model Prediction Results')
    plt.legend()
    plt.savefig('images/MambaModel.png', dpi=1000)
    plt.show()

    # 计算误差评估指标
    y_test = combined_results[:, 0]
    y_test_predict = combined_results[:, 1]
    R2 = 1 - np.sum((y_test - y_test_predict) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    MAE = mean_absolute_error(y_test_predict, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test_predict, y_test))
    MAPE = np.mean(np.abs((y_test - y_test_predict) / y_test)) * 100

    print(f'MAE: {MAE}, RMSE: {RMSE}, MAPE: {MAPE}, R2: {R2}')

    savef = pd.DataFrame({
        'MAE': [MAE],
        'RMSE': [RMSE],
        'MAPE': [MAPE],
        'R2': [R2]
    })
    savef.to_csv('results/error_MambaModel_i7_30.csv', index=False)
