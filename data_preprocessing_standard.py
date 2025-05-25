"""
===============================================================================
Title:        Standard Time-Series Preprocessing for Aquaculture Water Quality
File:         data_preprocessing_standard.py
Description:  This script performs standardized preprocessing for multivariate
              aquaculture water-quality time-series data. It supports:

              - Timestamp conversion and alignment
              - Missing value interpolation
              - Min-Max normalization (scaler saved)
              - Sliding window sample generation (X→Y)
              - Train/validation/test split
              - Torch DataLoader output for modeling

Input:        - CSV file named 'data.csv' containing:
                  ['Date', and multiple water quality indicators such as pH,
                   water temperature (WT), ORP, salinity, conductivity, air pressure, etc.,
                   and the target variable: dissolved oxygen (DO)']

Output:       - Processed CSV: processed/data_preprocessed.csv
              - Scaler: scaler/data_scaler.save
              - PyTorch DataLoaders (in memory): train_loader, val_loader, test_loader

Usage:
    1. Place your raw CSV file in the same directory and name it 'data.csv'
    2. Run the script using Python >= 3.8:
        $ python data_preprocessing_standard.py

    3. After running, you can directly use `train_loader`, etc. in your model training.

Environment:
    - Python >= 3.8
    - PyTorch >= 1.10
    - pandas
    - numpy
    - scikit-learn
    - joblib

Installation (via pip):
    pip install pandas numpy scikit-learn joblib torch


===============================================================================
"""



import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader

# ---------- Configuration ----------
file_path = 'data.csv'             # Raw data file
save_dir = 'processed'             # Folder for processed outputs
scaler_dir = 'scaler'              # Folder to save the scaler
seq_len = 30                       # Input sequence length
pred_len = 30                      # Prediction horizon
val_ratio = 0.1                    # Validation split ratio
test_ratio = 0.1                   # Test split ratio
batch_size = 128                   # Batch size for DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Step 1: Load and clean data ----------
df = pd.read_csv(file_path)

# Convert 'Date' to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Fill missing values using forward fill and linear interpolation
df.fillna(method='ffill', inplace=True)
df.interpolate(method='linear', inplace=True)


# ---------- Step 2: Normalize features ----------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
data_scaled_df = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)

# Create output directories
os.makedirs(save_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)

# Save preprocessed data
data_scaled_df.to_csv(os.path.join(save_dir, 'data_preprocessed.csv'))

# Save fitted scaler for inverse transform
joblib.dump(scaler, os.path.join(scaler_dir, 'data_scaler.save'))


# ---------- Step 3: Create sliding windows ----------
def create_sequences(data, seq_len, pred_len):
    """
    Convert time-series data into (X, Y) pairs using a sliding window.
    X shape: [seq_len, n_features], Y shape: [pred_len] (DO only)
    """
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x_seq = data[i:i + seq_len]
        y_seq = data[i + seq_len:i + seq_len + pred_len, -1]  # DO column
        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)


X, Y = create_sequences(data_scaled, seq_len, pred_len)


# ---------- Step 4: Split into train/val/test ----------
num_samples = len(X)
num_val = int(num_samples * val_ratio)
num_test = int(num_samples * test_ratio)
num_train = num_samples - num_val - num_test

X_train, Y_train = X[:num_train], Y[:num_train]
X_val, Y_val = X[num_train:num_train + num_val], Y[num_train:num_train + num_val]
X_test, Y_test = X[num_train + num_val:], Y[num_train + num_val:]

# Convert to PyTorch tensors and move to device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

# ---------- Step 5: Create DataLoaders ----------
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)


# ---------- Optional: Print summary ----------
print("✅ Data preprocessing completed.")
print(f"Total samples: {num_samples}")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Input shape: {X_train.shape}, Output shape: {Y_train.shape}")
print(f"Saved preprocessed data to: {os.path.join(save_dir, 'data_preprocessed.csv')}")
print(f"Saved scaler to: {os.path.join(scaler_dir, 'data_scaler.save')}")
