"""
===============================================================================
Title:        Time-Series Preprocessing with Periodic Features & Derived Metrics
File:         data_preprocessing_cycle_enhanced.py
Description:  This script performs enhanced preprocessing for aquaculture
              water-quality datasets with periodic or trend-sensitive variables
              such as DO, temperature, and pH.

              It supports:
              - Timestamp conversion and periodic feature extraction (hour, sin/cos)
              - Derived features (DO_diff, moving averages, salinity ratio)
              - Outlier detection and smoothing (MAD-based)
              - Feature scaling (MinMaxScaler)
              - Sliding window sequence generation
              - Train/validation/test split
              - PyTorch DataLoader output for deep models

Input:        - CSV file named 'data.csv' containing:
                  ['Date', and multiple water quality indicators such as pH,
                   water temperature (WT), ORP, salinity, conductivity, air pressure, etc.,
                   and the target variable: dissolved oxygen (DO)']

Output:       - Processed CSV: processed/data_cycle_enhanced.csv
              - Scaler: scaler/data_scaler_cycle.save
              - PyTorch DataLoaders: train_loader, val_loader, test_loader

Usage:
    1. Ensure 'data.csv' exists in the working directory
    2. Run using Python >= 3.8:
        $ python data_preprocessing_cycle_enhanced.py

    3. After running, load train_loader for model input

Environment:
    - Python >= 3.8
    - pandas, numpy, scikit-learn, joblib, torch

Installation:
    pip install pandas numpy scikit-learn joblib torch

===============================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch

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

# ---------- 1. Load raw data ----------
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.fillna(method='ffill', inplace=True)
df.interpolate(method='linear', inplace=True)

# ---------- 2. Extract periodic features ----------
df['hour'] = df.index.hour + df.index.minute / 60
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

# ---------- 3. Derived features ----------
df['DO_diff'] = df['DO'].diff().fillna(0)
df['DO_ma5'] = df['DO'].rolling(window=5, min_periods=1).mean()
df['WT_ma5'] = df['WT'].rolling(window=5, min_periods=1).mean()
df['Salinity_ratio'] = df['Salinity'] / (df['WT'] + 1e-5)  # 防止除以0

# ---------- 4. Outlier detection using MAD ----------
def mad_smooth(series, threshold=3):
    rolling_median = series.rolling(window=5, center=True).median()
    mad = (series - rolling_median).abs().rolling(window=5, center=True).median()
    outliers = (series - rolling_median).abs() > threshold * mad
    smoothed = series.copy()
    smoothed[outliers] = rolling_median[outliers]
    return smoothed

df['ORP'] = mad_smooth(df['ORP'])
df['Conductivity'] = mad_smooth(df['Conductivity'])

# ---------- 5. Normalize features ----------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
df_scaled = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)

# Save processed data and scaler
os.makedirs(save_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)
df_scaled.to_csv(os.path.join(save_dir, 'data_cycle_enhanced.csv'))
joblib.dump(scaler, os.path.join(scaler_dir, 'data_scaler_cycle.save'))

# ---------- 6. Sliding window construction ----------
def create_sequences(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x_seq = data[i:i + seq_len]
        y_seq = data[i + seq_len:i + seq_len + pred_len, df.columns.get_loc('DO')]  # target DO
        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)

X, Y = create_sequences(df_scaled.values, seq_len, pred_len)

# ---------- 7. Dataset split ----------
num_samples = len(X)
num_val = int(num_samples * val_ratio)
num_test = int(num_samples * test_ratio)
num_train = num_samples - num_val - num_test

X_train, Y_train = X[:num_train], Y[:num_train]
X_val, Y_val = X[num_train:num_train + num_val], Y[num_train:num_train + num_val]
X_test, Y_test = X[num_train + num_val:], Y[num_train + num_val:]

# ---------- 8. Torch Dataset creation ----------
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

# ---------- 9. Summary print ----------
print("✅ Cycle-enhanced preprocessing complete.")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Saved processed data to: {os.path.join(save_dir, 'data_cycle_enhanced.csv')}")
