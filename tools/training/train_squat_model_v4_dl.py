# -*- coding: utf-8 -*-
"""
Train Squat Model v4 — Deep Learning & SVM Explorations
Techniques:
  1. Multi-Layer Perceptron (Neural Network classifiers)
  2. Support Vector Machines (SVM)
  3. Feature Scaling (Crucial for NN and SVM)

Run: python tools/training/train_squat_model_v4_dl.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Deep Learning and SVM
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

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

LABEL_MAP = {0: 'Correct', 1: 'Shallow', 2: 'Forward Lean',
             3: 'Knees Caving', 4: 'Heels Off', 5: 'Asymmetric'}

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

def evaluate_models(X_train, X_test, y_train, y_test, title):
    print("\n" + "="*80)
    print(f"[{title}] EVALUATING SVM & NEURAL NETWORKS")
    print("="*80)
    
    # ALWAYS SCALE DATA for NN and SVM!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Support Vector Machine (RBF Kernel)
    print("\n1. Training Support Vector Machine (SVM - RBF Kernel)...")
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    
    y_pred_svm = svm_clf.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro')
    print(f"--> SVM Accuracy: {acc_svm:.4f} | F1: {f1_svm:.4f}")
    
    # 2. Shallow Neural Network (1 Hidden Layer)
    print("\n2. Training Shallow Neural Net (MLP - 1 Hidden Layer of 128 neurons)...")
    mlp_shallow = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, alpha=0.001,
                                solver='adam', random_state=42, early_stopping=True)
    mlp_shallow.fit(X_train_scaled, y_train)
    
    y_pred_mlp_s = mlp_shallow.predict(X_test_scaled)
    acc_mlp_s = accuracy_score(y_test, y_pred_mlp_s)
    f1_mlp_s = f1_score(y_test, y_pred_mlp_s, average='macro')
    print(f"--> Shallow NN Accuracy: {acc_mlp_s:.4f} | F1: {f1_mlp_s:.4f}")
    
    # 3. Deep Neural Network (3 Hidden Layers)
    print("\n3. Training Deep Neural Net (MLP - 3 Hidden Layers: 256 -> 128 -> 64)...")
    mlp_deep = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, alpha=0.001,
                             solver='adam', random_state=42, early_stopping=True)
    mlp_deep.fit(X_train_scaled, y_train)
    
    y_pred_mlp_d = mlp_deep.predict(X_test_scaled)
    acc_mlp_d = accuracy_score(y_test, y_pred_mlp_d)
    f1_mlp_d = f1_score(y_test, y_pred_mlp_d, average='macro')
    print(f"--> Deep NN Accuracy: {acc_mlp_d:.4f} | F1: {f1_mlp_d:.4f}")

    print("\n[SUMMARY] " + title)
    print(f"  SVM (RBF):           Acc={acc_svm:.4f},  F1={f1_svm:.4f}")
    print(f"  Shallow Neural Net:  Acc={acc_mlp_s:.4f},  F1={f1_mlp_s:.4f}")
    print(f"  Deep Neural Net:     Acc={acc_mlp_d:.4f},  F1={f1_mlp_d:.4f}")


def main():
    df = load_data()
    feat_cols = get_feature_cols(df)
    
    # ------------- BINARY CLASSIFICATION -------------
    X = df[feat_cols].fillna(0).values
    y_bin = (df['label'] != 0).astype(int)
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
        
    sm = SMOTE(random_state=42)
    X_res_b, y_res_b = sm.fit_resample(X_train_b, y_train_b)
    
    evaluate_models(X_res_b, X_test_b, y_res_b, y_test_b, "BINARY SQUAT FORM (Correct/Incorrect)")

    # ------------- MULTICLASS CLASSIFICATION -------------
    y_multi = df['label'].values
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
        
    X_res_m, y_res_m = sm.fit_resample(X_train_m, y_train_m)
    
    evaluate_models(X_res_m, X_test_m, y_res_m, y_test_m, "MULTICLASS SQUAT FORM (6 Errors)")

if __name__ == "__main__":
    main()
