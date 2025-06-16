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

# 定义 TCN 模块
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.3):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=1, padding=(kernel_size - 1) * dilation_size,
                                dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 定义 TCN + GRU 模型,,,Best hyperparameters:  {'gru_hidden_size': 64, 'tcn_channels': [64, 128, 256], 'dropout': 0.14852516619379708, 'lr': 0.0008410209482405987}
class TCN_GRU_Model(nn.Module):
    def __init__(self, n_features, gru_hidden_size, tcn_channels, length_size):
        super(TCN_GRU_Model, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=n_features, num_channels=tcn_channels)
        self.gru = nn.GRU(input_size=tcn_channels[-1], hidden_size=gru_hidden_size, num_layers=2, batch_first=True, dropout=0.1485)
                                                                                                                    #dropout = 0.2)
        self.fc = nn.Linear(gru_hidden_size, length_size)

    def forward(self, x):
        # 先通过 TCN 层
        x = x.transpose(1, 2)  # 调整输入形状以适应 TCN: [batch_size, n_features, seq_len]
        tcn_out = self.tcn(x)
        tcn_out = tcn_out.transpose(1, 2)  # 调整回 RNN 需要的形状: [batch_size, seq_len, tcn_channels[-1]]

        # 然后通过 GRU 层
        gru_out, _ = self.gru(tcn_out)
        output = self.fc(gru_out[:, -1, :])  # 取最后一个时间步的输出
        return output

# 设置参数
file_path = 'data\FC\FC3_I5.csv'
w = 60  # 模型输入序列长度
length_size = 60  # 预测结果的序列长度
e = 400  # 迭代次数 (增加迭代次数)
batch_size = 256  # 批量大小

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


# 模型训练#原始gru_hidden_size=128, tcn_channels=[32, 64, 128]
def model_train():
    net = TCN_GRU_Model(n_features=n_feature, gru_hidden_size=64, tcn_channels=[64, 128, 256], length_size=length_size).to(device)
    # 使用相同的gru_hidden_size和tcn_channels参数进行训练和测试
    criterion = nn.MSELoss().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=0.0008)

    best_val_loss = float('inf')
    best_model_path = 'checkpoint/best_TCN_GRU_Model_i5_30.pt'

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
    # 使用相同的模型架构定义
    net = TCN_GRU_Model(n_features=n_feature, gru_hidden_size=64, tcn_channels=[64, 128, 256], length_size=length_size).to(device)
    net.load_state_dict(torch.load('checkpoint/best_TCN_GRU_Model_i5_30.pt'))
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

    np.savetxt('true_final_pred_values_TCN_GRU_i5_30.csv', combined_results, delimiter=',', fmt='%.6f')

    # 可视化最终的结果
    time = np.arange(len(combined_results))
    plt.figure(figsize=(12, 3))
    plt.plot(time, combined_results[:, 0], c='red', linestyle='-', linewidth=1, label='True DO')
    plt.plot(time, combined_results[:, 1], c='blue', linestyle='--', linewidth=1, label='Predicted DO')
    plt.title('TCN + GRU Model Results')
    plt.legend()
    plt.savefig('images\TCN_GRU.png', dpi=100)
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
    savef.to_csv('results\error_TCN_GRU_i5_30.csv', index=False)
