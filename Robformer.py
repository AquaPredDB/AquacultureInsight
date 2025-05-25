import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义趋势与季节性分解模块
class RobustDecomposition(nn.Module):
    def __init__(self, in_channels):
        super(RobustDecomposition, self).__init__()
        self.trend_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.seasonal_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, x):
        trend = self.trend_conv(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return trend, seasonal

# 定义季节性调整模块
class SeasonalAdjustmentBlock(nn.Module):
    def __init__(self, in_channels):
        super(SeasonalAdjustmentBlock, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        return x * self.weight

# 定义Robformer结构
class Robformer(nn.Module):
    def __init__(self, n_features, length_size, embed_dim=32, num_heads=8, ff_dim=32, num_layers=1, dropout=0.31):
        super(Robformer, self).__init__()
        self.decomp = RobustDecomposition(n_features)
        self.seasonal_adjust = SeasonalAdjustmentBlock(n_features)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.input_projection = nn.Linear(n_features, embed_dim)
        self.fc = nn.Linear(embed_dim, length_size)

    def forward(self, x):
        trend, seasonal = self.decomp(x)
        seasonal = self.seasonal_adjust(seasonal)
        trend_encoded = self.transformer_encoder(self.input_projection(trend))
        output = self.fc(trend_encoded[:, -1, :])
        return output

# 设置参数    Best hyperparameters:  {'embed_dim': 32, 'num_heads': 8, 'ff_dim': 32, 'num_layers': 1, 'dropout': 0.31045305812380597, 'lr': 0.002831764258231043}
file_path = 'data\\FC\\FC3_I5.csv'
w = 30  # 模型输入序列长度
length_size = 30  # 预测结果的序列长度
e = 20  # 迭代次数
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
    net = Robformer(n_features=n_feature, length_size=length_size).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0028)

    best_val_loss = float('inf')
    best_model_path = 'checkpoint/best_Robformer_i5_30.pt'

    for epoch in range(e):
        net.train()
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            preds = net(datapoints)
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
                loss = criterion(preds, labels)
                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        print(f"Epoch: {epoch + 1}/{e}, Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), best_model_path)

    print("Training complete. Best validation loss: {:.6f}".format(best_val_loss))
    return net

# 模型测试
def model_test():
    net = Robformer(n_features=n_feature, length_size=length_size).to(device)
    net.load_state_dict(torch.load('checkpoint/best_Robformer_i5_30.pt'))
    net.eval()
    with torch.no_grad():
        X_test_gpu = X_test.to(device)
        pred = net(X_test_gpu)
        pred = pred.detach().cpu().numpy()
        true = y_test.detach().cpu().numpy()

    target_scaler = preprocessing.MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[-1:], scaler.scale_[-1:]

    pred_uninverse = target_scaler.inverse_transform(pred)
    true_uninverse = target_scaler.inverse_transform(true)

    return true_uninverse, pred_uninverse

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

if __name__ == "__main__":
    # 保持原有的模型训练部分不变
    net = model_train()
    true, pred = model_test()

    combined_results = np.column_stack((true[:, 0], pred[:, 0]))

    np.savetxt('true_final_pred_values_Robformer_i5_30.csv', combined_results, delimiter=',', fmt='%.6f')

    # 可视化最终的结果
    time = np.arange(len(combined_results))
    plt.figure(figsize=(12, 3))
    plt.plot(time, combined_results[:, 0], c='red', linestyle='-', linewidth=1, label='True DO')
    plt.plot(time, combined_results[:, 1], c='blue', linestyle='--', linewidth=1, label='Predicted DO')
    plt.title('Robformer Model Results')
    plt.legend()
    plt.savefig('images/Robformer.png', dpi=1000)
    plt.show()

    # 计算误差评估指标
    y_test = combined_results[:, 0]
    y_test_predict = combined_results[:, 1]
    R2 = 1 - np.sum((y_test - y_test_predict) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    MAE = mean_absolute_error(y_test_predict, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test_predict, y_test))
    MAPE = mape(y_test_predict, y_test)

    print(f'MAE: {MAE}, RMSE: {RMSE}, MAPE: {MAPE}, R2: {R2}')

    # 保存结果
    savef = pd.DataFrame({
        'MAE': [MAE],
        'RMSE': [RMSE],
        'MAPE': [MAPE],
        'R2': [R2]
    })
    savef.to_csv('results/error_Robformer_i5_30.csv', index=False)
