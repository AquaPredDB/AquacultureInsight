"""
===============================================================================
Title:        Trend-Residual Decomposition Preprocessing for Water Quality Forecasting
File:         data_preprocessing_trend_residual.py

Description:
    This script preprocesses dissolved oxygen (DO) time-series data by decomposing
    it into a trend component (via Savitzky-Golay smoothing) and a residual component.
    The residual is used as the prediction target, which enables the model to focus
    on short-term fluctuations while the trend is restored post-prediction.

Core Functionalities:
    - Timestamp parsing and alignment
    - Savitzky-Golay filtering to obtain DO trend
    - Residual computation: DO_residual = DO_original - DO_trend
    - Sliding window construction for trend + residual dual-channel input
    - Torch DataLoader creation
    - Save decomposed data and scalers

Input:
    - data.csv (columns: 'Date', 'DO', ...)

Output:
    - processed/data_trend_residual.csv
    - scaler/data_scaler_trend.save
    - PyTorch DataLoaders: train_loader, val_loader, test_loader
    - Input format: X: [N, T, 2], Y: [N, pred_len] (residual)

Usage:
    1. Ensure data.csv is in the current folder
    2. Run the script:
        $ python data_preprocessing_trend_residual.py

    3. In your model:
        DO_pred = DO_trend + predicted_residual

Dependencies:
    - pandas, numpy, scikit-learn, joblib, torch, scipy

Installation:
    pip install pandas numpy scikit-learn joblib torch scipy

Author:       [Your Name]
Date:         [YYYY-MM-DD]
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from torch.utils.data import TensorDataset, DataLoader
import torch
from scipy.signal import savgol_filter

# ---------- Config ----------
file_path = 'data.csv'
save_dir = 'processed'
scaler_dir = 'scaler'
seq_len = 30
pred_len = 30
val_ratio = 0.1
test_ratio = 0.1
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. Load data ----------
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.fillna(method='ffill', inplace=True)
df.interpolate(method='linear', inplace=True)

# ---------- 2. Smooth DO trend ----------
if 'DO' not in df.columns:
    raise ValueError("Missing target variable 'DO'.")

window_length = 11 if len(df) > 11 else len(df) - (len(df) + 1) % 2
polyorder = 2 if window_length > 2 else 1
DO_trend = savgol_filter(df['DO'].values, window_length=window_length, polyorder=polyorder)

# ---------- 3. Compute residual ----------
DO_residual = df['DO'].values - DO_trend

# ---------- 4. Normalize trend and residual separately ----------
scaler_trend = MinMaxScaler()
scaler_res = MinMaxScaler()

DO_trend_scaled = scaler_trend.fit_transform(DO_trend.reshape(-1, 1))
DO_residual_scaled = scaler_res.fit_transform(DO_residual.reshape(-1, 1))

# Save scalers
os.makedirs(scaler_dir, exist_ok=True)
joblib.dump(scaler_trend, os.path.join(scaler_dir, 'data_scaler_trend.save'))
joblib.dump(scaler_res, os.path.join(scaler_dir, 'data_scaler_residual.save'))

# ---------- 5. Combine as dual-channel input ----------
# Shape: [T, 2] --> [DO_trend, DO_residual]
X_dual = np.concatenate([DO_trend_scaled, DO_residual_scaled], axis=1)

# Save full sequence for reference
os.makedirs(save_dir, exist_ok=True)
df_out = df.copy()
df_out['DO_trend'] = DO_trend
df_out['DO_residual'] = DO_residual
df_out['DO_trend_scaled'] = DO_trend_scaled
df_out['DO_residual_scaled'] = DO_residual_scaled
df_out.to_csv(os.path.join(save_dir, 'data_trend_residual.csv'))

# ---------- 6. Sliding window ----------
def create_trend_residual_sequences(x_dual, y_residual, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(x_dual) - seq_len - pred_len + 1):
        x_seq = x_dual[i:i + seq_len]  # shape [T, 2]
        y_seq = y_residual[i + seq_len:i + seq_len + pred_len]
        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)

X, Y = create_trend_residual_sequences(X_dual, DO_residual_scaled.flatten(), seq_len, pred_len)

# ---------- 7. Split ----------
num_samples = len(X)
num_val = int(num_samples * val_ratio)
num_test = int(num_samples * test_ratio)
num_train = num_samples - num_val - num_test

X_train, Y_train = X[:num_train], Y[:num_train]
X_val, Y_val = X[num_train:num_train + num_val], Y[num_train:num_train + num_val]
X_test, Y_test = X[num_train + num_val:], Y[num_train + num_val:]

# ---------- 8. Torch format ----------
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

# ---------- 9. Summary ----------
print("✅ Trend-residual decomposition preprocessing completed.")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Input shape (X): {X_train.shape}, Output shape (Y): {Y_train.shape}")
print(f"Saved to: {os.path.join(save_dir, 'data_trend_residual.csv')}")
