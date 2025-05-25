"""
===============================================================================
Title:        Frequency-Domain Feature Enhanced Preprocessing for Aquaculture
File:         data_preprocessing_fft_enhanced.py

Description:
    This script enhances standard time-series input by extracting FFT-based
    frequency-domain features from each sliding window. These features are
    concatenated with time-domain variables, enabling models to learn
    both temporal and spectral patterns.

Key Functionalities:
    - Timestamp alignment & cleaning
    - FFT on each window to extract:
        - dominant frequency index
        - peak amplitude
        - average spectral energy
    - Normalization of FFT features
    - Concatenation of time + frequency domain features
    - PyTorch DataLoader output

Input:
    - data.csv: Must contain 'Date', 'DO', and other variables

Output:
    - processed/data_fft_enhanced.csv
    - scaler/data_scaler_fft.save
    - train_loader, val_loader, test_loader

Usage:
    1. Place your CSV as 'data.csv'
    2. Run: python data_preprocessing_fft_enhanced.py
    3. Use (X, Y) with frequency-augmented inputs for time series forecasting

Environment:
    - Python >= 3.8
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
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.fft import fft

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
    raise ValueError("Column 'DO' is required in the dataset.")

# ---------- 2. Normalize time-domain variables ----------
scaler_time = MinMaxScaler()
time_features = scaler_time.fit_transform(df.values)

# Save time scaler
os.makedirs(save_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)
joblib.dump(scaler_time, os.path.join(scaler_dir, 'data_scaler_fft.save'))

# ---------- 3. Sliding window + FFT extraction ----------
def compute_fft_features(window):
    """
    window: [T, F]
    Returns: [dominant_freq_idx, peak_amplitude, avg_energy] per feature
    """
    fft_feats = []
    for i in range(window.shape[1]):
        sig = window[:, i]
        freq_spec = np.abs(fft(sig))
        dominant_idx = np.argmax(freq_spec[:len(sig)//2])
        peak_amp = np.max(freq_spec)
        avg_energy = np.mean(freq_spec)
        fft_feats.append([dominant_idx / len(sig), peak_amp, avg_energy])
    return np.concatenate(fft_feats)

X_seq, Y_seq = [], []

for i in range(len(time_features) - seq_len - pred_len + 1):
    x_window = time_features[i:i+seq_len]  # shape: [seq_len, F]
    y_future = df['DO'].values[i+seq_len:i+seq_len+pred_len]  # original values

    # FFT features from this window
    fft_vector = compute_fft_features(x_window)  # shape: [F * 3]
    fft_vector = fft_vector.reshape(1, -1).repeat(seq_len, axis=0)  # repeat per timestep

    # Combine time + freq
    x_combined = np.concatenate([x_window, fft_vector], axis=1)
    X_seq.append(x_combined)
    Y_seq.append(y_future)

X = np.array(X_seq)  # [N, T, time+freq]
Y = np.array(Y_seq)  # [N, pred_len]

# Normalize FFT-only part (last N columns)
n_time_features = df.shape[1]
X_time = X[:, :, :n_time_features]
X_fft = X[:, :, n_time_features:]

scaler_fft = MinMaxScaler()
X_fft_flat = X_fft.reshape(-1, X_fft.shape[-1])
X_fft_scaled = scaler_fft.fit_transform(X_fft_flat)
X_fft = X_fft_scaled.reshape(X_fft.shape)

# Save scaler
joblib.dump(scaler_fft, os.path.join(scaler_dir, 'data_scaler_fft_features.save'))

# Combine back
X_final = np.concatenate([X_time, X_fft], axis=-1)  # [N, T, F + F_freq]

# ---------- 4. Train/val/test split ----------
num_samples = len(X_final)
num_val = int(num_samples * val_ratio)
num_test = int(num_samples * test_ratio)
num_train = num_samples - num_val - num_test

X_train, Y_train = X_final[:num_train], Y[:num_train]
X_val, Y_val = X_final[num_train:num_train + num_val], Y[num_train:num_train + num_val]
X_test, Y_test = X_final[num_train + num_val:], Y[num_train + num_val:]

# ---------- 5. Torch format ----------
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

# ---------- 6. Save and Print ----------
print("✅ FFT-enhanced preprocessing completed.")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Input shape: {X_train.shape} | Output shape: {Y_train.shape}")
print(f"Time features: {n_time_features}, FFT features: {X_train.shape[-1] - n_time_features}")
