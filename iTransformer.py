import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义 iTransformer 相关组件
class InvertedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(InvertedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, variate_num, _ = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into heads
        Q = Q.view(batch_size, variate_num, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, variate_num, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, variate_num, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, variate_num, self.d_model)

        # Final linear projection
        output = self.out_proj(attn_output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dim_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class InvertedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff):
        super(InvertedTransformerBlock, self).__init__()
        self.attention = InvertedSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dim_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.attention(x)
        x = x + attn_output
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)

        return x


class iTransformer(nn.Module):
    def __init__(self, input_len, pred_len, variate_num, d_model, num_heads, dim_ff, num_layers):
        super(iTransformer, self).__init__()
        # 嵌入层：将每个时间步的特征维度从 variate_num 映射到 d_model
        self.embedding = nn.Linear(input_len, d_model)  # 这里将 input_len 作为输入特征的维度

        self.transformer_blocks = nn.ModuleList([
            InvertedTransformerBlock(d_model, num_heads, dim_ff) for _ in range(num_layers)
        ])
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # 输入 x 的形状: [batch_size, variate_num, input_len]
        batch_size, variate_num, input_len = x.size()

        # 转置 x，使得 input_len 成为第二维度，特征数 variate_num 成为最后一维
        x = x.permute(0, 2, 1)  # [batch_size, input_len, variate_num]

        # 将每个时间步的 variate_num 特征嵌入到 d_model 维度
        x = self.embedding(x)  # [batch_size, input_len, d_model]

        # 通过 Transformer 块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # 形状保持不变: [batch_size, input_len, d_model]

        # 通过投影层，将 d_model 映射到预测长度 pred_len
        x = self.projection(x)  # 输出形状: [batch_size, input_len, pred_len]

        # 取最后一个时间步的输出作为最终预测
        x = x[:, -1, :]  # [batch_size, pred_len]

        return x


# 参数设置和数据读取保持不变
file_path = 'data/FC/FC3_I5.csv'
w = 120
length_size = 120
e = 200
batch_size = 256

data = pd.read_csv(file_path)
data = data.iloc[:, 1:]
data_dim = data.shape[1]
scaler = preprocessing.MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data_length = len(data_scaled)

train_size = int(0.8 * data_length)
val_size = int(0.1 * data_length)
test_size = data_length - train_size - val_size

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


# 模型训练和测试函数
def model_train():
    net = iTransformer(input_len=w, pred_len=length_size, variate_num=n_feature, d_model=64, num_heads=4, dim_ff=128,
                       num_layers=3).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_val_loss = float('inf')
    best_model_path = 'checkpoint/best_iTransformer_I5_30.pt'

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


# 模型测试部分
def model_test():
    net = iTransformer(input_len=w, pred_len=length_size, variate_num=n_feature, d_model=64, num_heads=4, dim_ff=128,
                       num_layers=3).to(device)
    net.load_state_dict(torch.load('checkpoint/best_iTransformer_I5_30.pt'))
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


# 主程序
if __name__ == "__main__":
    net = model_train()
    true, pred = model_test()

    combined_results = np.column_stack((true[:, 0], pred[:, 0]))

    np.savetxt('true_pred_values_iTransformer_I5_30.csv', combined_results, delimiter=',', fmt='%.6f')

    # 可视化
    time = np.arange(len(combined_results))
    plt.figure(figsize=(12, 3))
    plt.plot(time, combined_results[:, 0], c='red', linestyle='-', linewidth=1, label='True DO')
    plt.plot(time, combined_results[:, 1], c='blue', linestyle='--', linewidth=1, label='Predicted DO')
    plt.title('iTransformer Results')
    plt.legend()
    plt.savefig('images/iTransformer.png', dpi=1000)
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
    savef.to_csv('results/error_iTransformer_I5_30.csv', index=False)
