"""
===============================================================================
Title:        Smoothed Target & Anomaly Labeling Preprocessing for Aquaculture
File:         data_preprocessing_smooth_anomaly.py

Description:
    This script preprocesses high-frequency water-quality time series by applying
    light smoothing to the target variable (e.g., DO) and labeling high-frequency
    spikes as binary anomalies. It supports multi-task training setups where the
    model predicts both smoothed values and anomaly flags.

Key Functionalities:
    - Timestamp alignment and missing value interpolation
    - Apply moving average to smooth DO labels (anti-overfitting)
    - Detect short-term spikes and label as binary anomaly mask
    - Generate inputs and targets for dual-output models

Input:
    - data.csv: Must contain ['Date', 'DO', ...] with high-resolution sampling

Output:
    - processed/data_smooth_anomaly.csv
    - scaler/data_scaler_smooth.save
    - train_loader, val_loader, test_loader
    - Target:
        Y1 = smoothed DO
        Y2 = anomaly label (0/1)

Usage:
    1. Place data.csv in working directory
    2. Run script:
        $ python data_preprocessing_smooth_anomaly.py

    3. In model training:
        Main task: predict smoothed DO
        Aux task: predict anomaly (binary mask)

Environment:
    - Python >= 3.8
    - pandas, numpy, scikit-learn, joblib, torch

Installation:
    pip install pandas numpy scikit-learn joblib torch

Author:       [Your Name]
Date:         [YYYY-MM-DD]
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

# ---------- 1. Load and clean ----------
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.fillna(method='ffill', inplace=True)
df.interpolate(method='linear', inplace=True)

if 'DO' not in df.columns:
    raise ValueError("DO column missing.")

# ---------- 2. Apply moving average smoothing (label smoothing) ----------
window = 5  # Light smoothing
DO_raw = df['DO'].values
DO_smooth = pd.Series(DO_raw).rolling(window=window, center=True, min_periods=1).mean().values

# ---------- 3. Detect high-frequency anomalies ----------
# Define anomaly as point deviating too much from smoothed version
residual = np.abs(DO_raw - DO_smooth)
threshold = 2 * np.std(residual)  # Dynamic threshold
anomaly_mask = (residual > threshold).astype(int)

# ---------- 4. Normalize features ----------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
df_scaled = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)

# Save outputs
os.makedirs(save_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(scaler_dir, 'data_scaler_smooth.save'))

# Add smoothed DO and anomaly to DataFrame for inspection
df_out = df.copy()
df_out['DO_smooth'] = DO_smooth
df_out['anomaly'] = anomaly_mask
df_out.to_csv(os.path.join(save_dir, 'data_smooth_anomaly.csv'))

# ---------- 5. Sliding window with dual targets ----------
def create_dual_target_sequences(data_X, DO_smooth, anomaly, seq_len, pred_len):
    X, Y_smooth, Y_anomaly = [], [], []
    for i in range(len(data_X) - seq_len - pred_len + 1):
        x_seq = data_X[i:i+seq_len]
        y1 = DO_smooth[i+seq_len:i+seq_len+pred_len]
        y2 = anomaly[i+seq_len:i+seq_len+pred_len]
        X.append(x_seq)
        Y_smooth.append(y1)
        Y_anomaly.append(y2)
    return np.array(X), np.array(Y_smooth), np.array(Y_anomaly)

X, Y1, Y2 = create_dual_target_sequences(df_scaled.values, DO_smooth, anomaly_mask, seq_len, pred_len)

# ---------- 6. Split ----------
num_samples = len(X)
num_val = int(num_samples * val_ratio)
num_test = int(num_samples * test_ratio)
num_train = num_samples - num_val - num_test

X_train, Y1_train, Y2_train = X[:num_train], Y1[:num_train], Y2[:num_train]
X_val, Y1_val, Y2_val = X[num_train:num_train+num_val], Y1[num_train:num_train+num_val], Y2[num_train:num_train+num_val]
X_test, Y1_test, Y2_test = X[num_train+num_val:], Y1[num_train+num_val:], Y2[num_train+num_val:]

# ---------- 7. Torch format ----------
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y1_train = torch.tensor(Y1_train, dtype=torch.float32).to(device)
Y2_train = torch.tensor(Y2_train, dtype=torch.float32).to(device)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
Y1_val = torch.tensor(Y1_val, dtype=torch.float32).to(device)
Y2_val = torch.tensor(Y2_val, dtype=torch.float32).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y1_test = torch.tensor(Y1_test, dtype=torch.float32).to(device)
Y2_test = torch.tensor(Y2_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y1_train, Y2_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y1_val, Y2_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y1_test, Y2_test), batch_size=batch_size, shuffle=False)

# ---------- 8. Summary ----------
print("✅ Smoothed DO and anomaly labeling preprocessing completed.")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"X shape: {X_train.shape}, Y1 (smooth): {Y1_train.shape}, Y2 (anomaly): {Y2_train.shape}")
print(f"Saved smoothed labels and anomaly mask to: {os.path.join(save_dir, 'data_smooth_anomaly.csv')}")
