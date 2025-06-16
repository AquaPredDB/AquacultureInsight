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


# 定义时间序列分解组件
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonality = x - trend
        return seasonality, trend


# 定义LayerNorm层
class my_Layernorm(nn.Module):
    def __init__(self, d_model):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layernorm(x)


# 定义AutoCorrelation层
class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag, factor, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / (E ** 0.5)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scores = self.dropout(torch.softmax(scale * scores, dim=-1))
        if self.output_attention:
            return torch.einsum("bhls,bshd->blhd", scores, values), scores
        else:
            return torch.einsum("bhls,bshd->blhd", scores, values), None


# 定义AutoCorrelationLayer层
class AutoCorrelationLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AutoCorrelationLayer, self).__init__()
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape

        queries = self.query_projection(queries).reshape(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).reshape(B, S, self.n_heads, -1)
        values = self.value_projection(values).reshape(B, S, self.n_heads, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask
        )

        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn


# 定义EncoderLayer层
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, moving_avg, dropout=0.1, activation="gelu"):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention_layer(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(new_x))

        y = self.conv1(x.permute(0, 2, 1))
        y = self.activation(y)
        y = self.conv2(y).permute(0, 2, 1)

        # 为了确保x和y具有相同的大小，在维度2上进行适当的填充
        if x.size(2) != y.size(2):
            y = torch.nn.functional.pad(y, (0, x.size(2) - y.size(2)))

        return self.norm2(x + self.dropout(y)), attn


# 定义Encoder层
class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


# 定义Autoformer模型
class Autoformer(nn.Module):
    def __init__(self, n_features, length_size, d_model=8, nhead=2, d_ff=64, e_layers=2, factor=5):
        super(Autoformer, self).__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.decomp = series_decomp(kernel_size=25)
        self.encoder = Encoder(
            [EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, factor, attention_dropout=0.1, output_attention=False),
                    d_model, nhead),
                d_model, d_ff, moving_avg=25, dropout=0.1, activation='gelu'
            ) for _ in range(e_layers)],
            norm_layer=my_Layernorm(d_model)
        )
        self.fc = nn.Linear(d_model, length_size)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.decomp(x)[0]  # 只提取趋势部分
        x, _ = self.encoder(x)
        x = self.fc(x[:, -1, :])
        return x


# 设置参数     
file_path = 'data\\FC\\FC3_I5.csv'
w = 30  # 模型输入序列长度
length_size = 30  # 预测结果的序列长度
e = 200  # 迭代次数
batch_size = 256  # 批量大小


# 读取数据
data = pd.read_csv(file_path)
data = data.iloc[:, 1:]  # 去除时间列
data_target = data.iloc[:, -1:]  # 目标数据
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

    X_train, y_train = torch.tensor(X_train).to(torch.float32).to(device), torch.tensor(y_train).to(torch.float32).to(
        device)
    ds = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dataloader, X_train, y_train


dataloader_train, X_train, y_train = data_loader(w, length_size, batch_size, data_train)
dataloader_val, X_val, y_val = data_loader(w, length_size, batch_size, data_val)
dataloader_test, X_test, y_test = data_loader(w, length_size, batch_size, data_test)


def model_train():
    net = Autoformer(n_features=n_feature, length_size=length_size).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_model_path = 'checkpoint/best_Autoformer_i5_30.pt'

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

        # 在每个epoch结束时评估验证集上的损失
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for datapoints, labels in dataloader_val:
                preds = net(datapoints)
                loss = criterion(preds, labels)
                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        print(f"Epoch: {epoch + 1}/{e}, Validation Loss: {val_loss:.6f}")

        # 保存验证损失最小的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), best_model_path)

    print("Training complete. Best validation loss: {:.6f}".format(best_val_loss))
    return net


def model_test():
    net = Autoformer(n_features=n_feature, length_size=length_size).to(device)
    net.load_state_dict(torch.load('checkpoint/best_Autoformer_i5_30.pt'))
    net.eval()
    with torch.no_grad():
        X_test_gpu = X_test.to(device)
        pred = net(X_test_gpu)
        pred = pred.detach().cpu().numpy()
        true = y_test.detach().cpu().numpy()

    # 为目标列（pred 和 true）创建一个新的 MinMaxScaler 实例
    target_scaler = preprocessing.MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[-1:], scaler.scale_[-1:]

    # 对预测值和真实值进行逆变换
    pred_uninverse = target_scaler.inverse_transform(pred)
    true_uninverse = target_scaler.inverse_transform(true)

    return true_uninverse, pred_uninverse


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


# 保持和显示真实值与预测值的代码
if __name__ == "__main__":
    model_train()
    true, pred = model_test()

    # 只保留溶解氧的真实值和预测值
    combined_results = np.column_stack((true[:, 0], pred[:, 0]))

    # 将结果保存到文件
    np.savetxt('true_pred_values_Autoformer_i5_30.csv', combined_results, delimiter=',', fmt='%.6f')

    time = np.arange(len(combined_results))
    plt.figure(figsize=(12, 3))
    plt.plot(time, combined_results[:, 0], c='red', linestyle='-', linewidth=1, label='True DO')
    plt.plot(time, combined_results[:, 1], c='black', linestyle='--', linewidth=1, label='Predicted DO')
    plt.title('Autoformer Prediction Results')
    plt.legend()
    plt.savefig('images\\Autoformer.png', dpi=1000)
    plt.show()

    y_test = combined_results[:, 0]
    y_test_predict = combined_results[:, 1]
    R2 = 1 - np.sum((y_test - y_test_predict) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    MAE = mean_absolute_error(y_test_predict, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test_predict, y_test))
    MAPE = mape(y_test_predict, y_test)
    print('MAE:', MAE)
    print('RMSE:', RMSE)
    print('MAPE:', MAPE)
    print('R2:', R2)

    savef = pd.DataFrame({
        'MAE': [MAE],
        'RMSE': [RMSE],
        'MAPE': [MAPE],
        'R2': [R2]
    })
    savef.to_csv('results\\error_Autoformer_i5_30.csv', index=False)
