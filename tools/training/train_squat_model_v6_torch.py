# -*- coding: utf-8 -*-
"""
Train Squat Model v6 — Deep Learning with PyTorch
Techniques:
  1. Deep Neural Network (PyTorch nn.Module)
  2. Adam Optimizer, CrossEntropyLoss/BCEWithLogitsLoss
  3. Feature Scaling & SMOTE

Run: python tools/training/train_squat_model_v6_torch.py
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent.parent))

KAGGLE_CSV = Path("C:/fit/FitVision/data/raw/kaggle_squat/squat_features_augmented.csv")

BASE_FEATURES = [
    'left_knee_angle',   'right_knee_angle',
    'left_hip_angle',    'right_hip_angle',
    'left_ankle_angle',  'right_ankle_angle',
    'spine_angle',       'torso_lean',
    'left_knee_lateral', 'right_knee_lateral',
    'symmetry_score',    'hip_depth',
]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['avg_knee_angle'] = (df['left_knee_angle'] + df['right_knee_angle']) / 2
    df['avg_hip_angle']  = (df['left_hip_angle']  + df['right_hip_angle'])  / 2
    df['knee_hip_ratio'] = df['avg_knee_angle'] / (df['avg_hip_angle'] + 1e-8)
    df['knee_depth_ratio'] = df['avg_knee_angle'] / 90.0
    df['ankle_asymmetry'] = (df['left_ankle_angle'] - df['right_ankle_angle']).abs()
    df['hip_asymmetry'] = (df['left_hip_angle'] - df['right_hip_angle']).abs()
    df['total_lateral'] = df['left_knee_lateral'].abs() + df['right_knee_lateral'].abs()
    df['lean_consistency'] = (df['spine_angle'] - df['torso_lean']).abs()
    return df

def get_feature_cols(df):
    eng_cols = [
        'avg_knee_angle', 'avg_hip_angle', 'knee_hip_ratio',
        'knee_depth_ratio', 'ankle_asymmetry', 'hip_asymmetry',
        'total_lateral', 'lean_consistency',
    ]
    return BASE_FEATURES + [c for c in eng_cols if c in df.columns]

def load_data():
    df = pd.read_csv(KAGGLE_CSV)
    df = add_engineered_features(df)
    return df

# Define the PyTorch Model Architecture
class SquatNet(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(SquatNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def train_model(model, dataloader, criterion, optimizer, epochs=30, device='cpu'):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Squeeze output for binary classification to match targets shape
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
                
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X_test, y_test, device='cpu', binary=True):
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test).to(device)
        outputs = model(inputs)
        
        if binary:
            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).cpu().numpy().astype(int)
        else:
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    return acc, f1

def main():
    print("\n" + "="*70)
    print("FITVISION DEEP LEARNING EXP. V6 (PYTORCH)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    df = load_data()
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(0).values
    
    # ── 1. BINARY CLASSIFICATION ──────────────────────────────────────────────
    print("\n[STAGE 1] Training Binary Model (Correct / Incorrect)...")
    y_bin = (df['label'] != 0).astype(int)
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
        
    sm = SMOTE(random_state=42)
    X_res_b, y_res_b = sm.fit_resample(X_train_b, y_train_b)
    
    scaler = StandardScaler()
    X_res_b_scaled = scaler.fit_transform(X_res_b)
    X_test_b_scaled = scaler.transform(X_test_b)
    
    train_dataset_b = TensorDataset(torch.FloatTensor(X_res_b_scaled), torch.FloatTensor(y_res_b))
    train_loader_b = DataLoader(train_dataset_b, batch_size=256, shuffle=True)
    
    model_bin = SquatNet(X_res_b_scaled.shape[1], num_classes=1).to(device)
    criterion_bin = nn.BCEWithLogitsLoss()
    optimizer_bin = optim.Adam(model_bin.parameters(), lr=0.001)
    
    print("Epochs running... (Showing progress implicitly without spamming)")
    train_model(model_bin, train_loader_b, criterion_bin, optimizer_bin, epochs=30, device=device)
    
    acc_b, f1_b = evaluate_model(model_bin, X_test_b_scaled, y_test_b, device=device, binary=True)
    print(f"--> [PYTORCH BINARY] Accuracy: {acc_b:.4f}  |  F1: {f1_b:.4f}")
    
    
    # ── 2. MULTICLASS CLASSIFICATION ──────────────────────────────────────────
    print("\n[STAGE 2] Training Multiclass Model (6 Specific Error Types)...")
    y_multi = df['label'].values
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
        
    X_res_m, y_res_m = sm.fit_resample(X_train_m, y_train_m)
    
    scaler_m = StandardScaler()
    X_res_m_scaled = scaler_m.fit_transform(X_res_m)
    X_test_m_scaled = scaler_m.transform(X_test_m)
    
    train_dataset_m = TensorDataset(torch.FloatTensor(X_res_m_scaled), torch.LongTensor(y_res_m))
    train_loader_m = DataLoader(train_dataset_m, batch_size=256, shuffle=True)
    
    # 6 classes
    model_multi = SquatNet(X_res_m_scaled.shape[1], num_classes=6).to(device)
    criterion_multi = nn.CrossEntropyLoss()
    optimizer_multi = optim.Adam(model_multi.parameters(), lr=0.001)
    
    print("Epochs running... (Showing progress implicitly without spamming)")
    train_model(model_multi, train_loader_m, criterion_multi, optimizer_multi, epochs=30, device=device)
    
    acc_m, f1_m = evaluate_model(model_multi, X_test_m_scaled, y_test_m, device=device, binary=False)
    print(f"--> [PYTORCH MULTICLASS] Accuracy: {acc_m:.4f}  |  F1: {f1_m:.4f}")

if __name__ == "__main__":
    main()
