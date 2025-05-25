"""
===============================================================================
Title:        Structured Group-Based Preprocessing for Aquaculture Water Quality
File:         data_preprocessing_structured_groups.py

Description:
    This script implements a preprocessing method that groups multivariate
    aquaculture time-series data into semantically meaningful categories (e.g.,
    biological vs. environmental variables), and prepares them as structured
    multi-channel sequences for models like multi-branch Transformers or
    group-based attention networks.

    Core functionalities:
    - Timestamp parsing and alignment
    - Feature grouping by domain knowledge
    - Independent MinMaxScaler normalization per group
    - Save group-specific scalers
    - Sequence generation with grouped structure: X.shape = [N, T, G, Fg]
    - Train/val/test splitting
    - PyTorch DataLoader output

Input:
    A CSV file named 'data.csv', with at least these columns:
        - 'Date'
        - Biological group: ['DO', 'pH', 'WT']
        - Environmental group: ['ORP', 'Salinity', 'Conductivity', 'Air Pressure']

Output:
    - Grouped & normalized sequence file: processed/data_structured_grouped.csv
    - Saved scalers: scaler/scaler_group_bio.save, scaler_group_env.save
    - PyTorch DataLoaders: train_loader, val_loader, test_loader (multi-channel)

Usage:
    1. Place 'data.csv' in the working directory
    2. Run the script:
        $ python data_preprocessing_structured_groups.py
    3. Use train_loader, val_loader, test_loader in your multi-channel model

Dependencies:
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

# ---------- 2. Group features ----------
# Define grouped feature names
bio_group = ['DO', 'pH', 'WT']
env_group = ['ORP', 'Salinity', 'Conductivity', 'Air Pressure']

# Check existence
for col in bio_group + env_group:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: {col}")

# Extract groups
df_bio = df[bio_group].copy()
df_env = df[env_group].copy()

# ---------- 3. Normalize each group independently ----------
scaler_bio = MinMaxScaler()
scaler_env = MinMaxScaler()

scaled_bio = scaler_bio.fit_transform(df_bio.values)
scaled_env = scaler_env.fit_transform(df_env.values)

# Save scalers
os.makedirs(save_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)
joblib.dump(scaler_bio, os.path.join(scaler_dir, 'scaler_group_bio.save'))
joblib.dump(scaler_env, os.path.join(scaler_dir, 'scaler_group_env.save'))

# ---------- 4. Recombine and save normalized DataFrame ----------
df_scaled = pd.DataFrame(
    np.concatenate([scaled_bio, scaled_env], axis=1),
    columns=bio_group + env_group,
    index=df.index
)
df_scaled.to_csv(os.path.join(save_dir, 'data_structured_grouped.csv'))

# ---------- 5. Construct multi-channel sequences ----------
def create_grouped_sequences(bio_array, env_array, seq_len, pred_len):
    """
    Returns:
        X: [N, T, G, Fg]  (e.g., [samples, time, 2, group_features])
        Y: [N, pred_len]  (target is future DO values)
    """
    X, Y = [], []
    for i in range(len(bio_array) - seq_len - pred_len + 1):
        bio_seq = bio_array[i:i+seq_len]
        env_seq = env_array[i:i+seq_len]
        y_seq = bio_array[i+seq_len:i+seq_len+pred_len, 0]  # DO is first in bio group
        grouped = np.stack([bio_seq, env_seq], axis=1)  # Shape: [T, G, Fg]
        X.append(grouped)
        Y.append(y_seq)
    return np.array(X), np.array(Y)

X, Y = create_grouped_sequences(scaled_bio, scaled_env, seq_len, pred_len)

# ---------- 6. Split ----------
num_samples = len(X)
num_val = int(num_samples * val_ratio)
num_test = int(num_samples * test_ratio)
num_train = num_samples - num_val - num_test

X_train, Y_train = X[:num_train], Y[:num_train]
X_val, Y_val = X[num_train:num_train + num_val], Y[num_train:num_train + num_val]
X_test, Y_test = X[num_train + num_val:], Y[num_train + num_val:]

# ---------- 7. Torch Dataset ----------
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

# ---------- 8. Summary ----------
print("✅ Group-structured preprocessing completed.")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Input shape (X): {X_train.shape} | Grouped as: [batch, time, group, group_features]")
print(f"Saved to: {os.path.join(save_dir, 'data_structured_grouped.csv')}")
