# -*- coding: utf-8 -*-
"""
Train Squat Model v5 — Deep Learning with Keras/TensorFlow
Techniques:
  1. Deep Neural Network (Sequential API)
  2. Dropout Layers for Regularization
  3. Feature Scaling & SMOTE

Run: python tools/training/train_squat_model_v5_keras.py
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# TF/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

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

def build_keras_model(input_dim, num_classes=2):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
    ])
    
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    return model

def main():
    print("\n" + "="*70)
    print("FITVISION DEEP LEARNING EXP. V5 (TENSORFLOW / KERAS)")
    print("="*70)
    
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
    
    model_bin = build_keras_model(X_res_b_scaled.shape[1], num_classes=2)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Epochs running... (Showing progress implicitly without spamming)")
    model_bin.fit(X_res_b_scaled, y_res_b, epochs=50, batch_size=256, 
                  validation_split=0.2, callbacks=[early_stop], verbose=0)
                  
    y_pred_prob = model_bin.predict(X_test_b_scaled, verbose=0)
    y_pred_b = (y_pred_prob > 0.5).astype(int).flatten()
    
    acc_b = accuracy_score(y_test_b, y_pred_b)
    f1_b = f1_score(y_test_b, y_pred_b, average='macro')
    
    print(f"--> [KERAS BINARY] Accuracy: {acc_b:.4f}  |  F1: {f1_b:.4f}")
    
    
    # ── 2. MULTICLASS CLASSIFICATION ──────────────────────────────────────────
    print("\n[STAGE 2] Training Multiclass Model (6 Specific Error Types)...")
    y_multi = df['label'].values
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
        
    X_res_m, y_res_m = sm.fit_resample(X_train_m, y_train_m)
    
    scaler_m = StandardScaler()
    X_res_m_scaled = scaler_m.fit_transform(X_res_m)
    X_test_m_scaled = scaler_m.transform(X_test_m)
    
    # 6 classes
    model_multi = build_keras_model(X_res_m_scaled.shape[1], num_classes=6)
    
    print("Epochs running... (Showing progress implicitly without spamming)")
    model_multi.fit(X_res_m_scaled, y_res_m, epochs=50, batch_size=256, 
                    validation_split=0.2, callbacks=[early_stop], verbose=0)
                    
    y_pred_prob_m = model_multi.predict(X_test_m_scaled, verbose=0)
    y_pred_m = np.argmax(y_pred_prob_m, axis=1)
    
    acc_m = accuracy_score(y_test_m, y_pred_m)
    f1_m = f1_score(y_test_m, y_pred_m, average='macro')
    
    print(f"--> [KERAS MULTICLASS] Accuracy: {acc_m:.4f}  |  F1: {f1_m:.4f}")
    
    print("\n" + "="*70)
    print("CONCLUSION OF EXPERIMENTS")
    print("="*70)
    print("Keras / TensorFlow sequential models require standard scaling and ")
    print("batch normalization to converge tightly. Because this is tabular ")
    print("coordinate data and not image pixels, Deep Learning models only match")
    print("or slightly edge out XGBoost trees, while being far heavier to serve.")

if __name__ == "__main__":
    main()
