import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import math
import os

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 原始位置编码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# 加入原始 Transformer Encoder + Decoder 架构
class TransformerWithDecoder(nn.Module):
    def __init__(self, n_features, length_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerWithDecoder, self).__init__()
        self.d_model = d_model
        self.length_size = length_size
        self.encoder_input_proj = nn.Linear(n_features, d_model)
        self.decoder_input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.encoder_input_proj(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src)

        tgt = self.decoder_input_proj(tgt)
        tgt = self.pos_decoder(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        out = self.output_proj(out)
        return out.squeeze(-1)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

# 设置参数
file_path = 'data/FC/FC3_I88.csv'
w = 120
length_size = 120
e = 600
batch_size = 256

os.makedirs('checkpoint', exist_ok=True)
os.makedirs('images', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 数据处理
data = pd.read_csv(file_path)
data = data.iloc[:, 1:]
data_target = data.iloc[:, -1:]
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
    sequence_length = w + length_size
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)
    x = result[:, :w, :]
    y = result[:, w:, -1]
    y_teacher = y[:, :-1]
    y_target = y[:, 1:]

    X = torch.tensor(x, dtype=torch.float32).to(device)
    Y_teacher = torch.tensor(y_teacher, dtype=torch.float32).unsqueeze(-1).to(device)
    Y_target = torch.tensor(y_target, dtype=torch.float32).to(device)

    ds = torch.utils.data.TensorDataset(X, Y_teacher, Y_target)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dataloader, X, Y_teacher, Y_target

dataloader_train, X_train, Y_teacher_train, Y_target_train = data_loader(w, length_size, batch_size, data_train)
dataloader_val, X_val, Y_teacher_val, Y_target_val = data_loader(w, length_size, batch_size, data_val)
dataloader_test, X_test, Y_teacher_test, Y_target_test = data_loader(w, length_size, batch_size, data_test)

def model_train():
    net = TransformerWithDecoder(n_features=n_feature, length_size=length_size).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    best_val_loss = float('inf')
    best_model_path = 'checkpoint/best_Transformer_Decoder_I8_120.pt'

    for epoch in range(e):
        net.train()
        for i, (src, tgt_in, tgt_out) in enumerate(dataloader_train):
            optimizer.zero_grad()
            preds = net(src, tgt_in)
            loss = criterion(preds, tgt_out)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch: {epoch+1}/{e}, Step: {i}, Training Loss: {loss.item():.6f}")

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt_in, tgt_out in dataloader_val:
                preds = net(src, tgt_in)
                loss = criterion(preds, tgt_out)
                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        print(f"Epoch: {epoch+1}/{e}, Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), best_model_path)

    print("Training complete. Best validation loss: {:.6f}".format(best_val_loss))
    return net

def model_test():
    net = TransformerWithDecoder(n_features=n_feature, length_size=length_size).to(device)
    net.load_state_dict(torch.load('checkpoint/best_Transformer_Decoder_I8_120.pt'))
    net.eval()
    with torch.no_grad():
        pred = net(X_test, Y_teacher_test)
        pred = pred.detach().cpu().numpy()
        true = Y_target_test.detach().cpu().numpy()

    target_scaler = preprocessing.MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[-1:], scaler.scale_[-1:]
    pred_uninverse = target_scaler.inverse_transform(pred)
    true_uninverse = target_scaler.inverse_transform(true)
    return true_uninverse, pred_uninverse

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6))) * 100

if __name__ == "__main__":
    model_train()
    true, pred = model_test()
    combined_results = np.column_stack((true[:, 0], pred[:, 0]))
    np.savetxt('true_pred_TransformerDecoder_I8_120.csv', combined_results, delimiter=',', fmt='%.6f')

    time = np.arange(len(combined_results))
    plt.figure(figsize=(12, 3))
    plt.plot(time, combined_results[:, 0], c='red', linestyle='-', linewidth=1, label='True DO')
    plt.plot(time, combined_results[:, 1], c='black', linestyle='--', linewidth=1, label='Predicted DO')
    plt.title('Transformer with Decoder Prediction Results')
    plt.legend()
    plt.savefig('images/TransformerDecoder.png', dpi=1000)
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
    savef.to_csv('results/error_TransformerDecoder_I8_120.csv', index=False)
