# -*- coding: utf-8 -*-
"""
Deadlift Model Experiments (XGBoost vs SVM vs MLP)
Run: python tools/training/train_deadlift_experiments.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Advanced Models
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent.parent))

DATA_CSV = Path("C:/fit/FitVision/data/processed/training_dataset.csv")

def evaluate_models(X_train, X_test, y_train, y_test):
    print("\n" + "="*80)
    print("EVALUATING DEADLIFT MODELS (Correct vs Incorrect)")
    print("="*80)
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 0. Random Forest (Baseline)
    print("\n0. Training Random Forest (Baseline)...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                    class_weight='balanced', random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    print(f"--> Random Forest Accuracy: {acc_rf:.4f} | F1: {f1_rf:.4f}")

    # 1. XGBoost
    print("\n1. Training XGBoost...")
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                eval_metric='logloss', use_label_encoder=False, 
                                random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_clf.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
    print(f"--> XGBoost Accuracy: {acc_xgb:.4f} | F1: {f1_xgb:.4f}")

    # 2. Support Vector Machine (RBF Kernel)
    print("\n2. Training Support Vector Machine (SVM - RBF Kernel)...")
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    y_pred_svm = svm_clf.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro')
    print(f"--> SVM Accuracy: {acc_svm:.4f} | F1: {f1_svm:.4f}")
    
    # 3. Shallow Neural Network (1 Hidden Layer)
    print("\n3. Training Shallow Neural Net (MLP - 1 Hidden Layer of 128 neurons)...")
    mlp_shallow = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, alpha=0.001,
                                solver='adam', random_state=42, early_stopping=True)
    mlp_shallow.fit(X_train_scaled, y_train)
    y_pred_mlp_s = mlp_shallow.predict(X_test_scaled)
    acc_mlp_s = accuracy_score(y_test, y_pred_mlp_s)
    f1_mlp_s = f1_score(y_test, y_pred_mlp_s, average='macro')
    print(f"--> Shallow NN Accuracy: {acc_mlp_s:.4f} | F1: {f1_mlp_s:.4f}")
    
    # 4. Deep Neural Network (3 Hidden Layers)
    print("\n4. Training Deep Neural Net (MLP - 3 Hidden Layers: 256 -> 128 -> 64)...")
    mlp_deep = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, alpha=0.001,
                             solver='adam', random_state=42, early_stopping=True)
    mlp_deep.fit(X_train_scaled, y_train)
    y_pred_mlp_d = mlp_deep.predict(X_test_scaled)
    acc_mlp_d = accuracy_score(y_test, y_pred_mlp_d)
    f1_mlp_d = f1_score(y_test, y_pred_mlp_d, average='macro')
    print(f"--> Deep NN Accuracy: {acc_mlp_d:.4f} | F1: {f1_mlp_d:.4f}")

    print("\n" + "="*80)
    print("[SUMMARY] DEADLIFT FORM (Correct/Incorrect)")
    print("="*80)
    print(f"  Random Forest (V1):  Acc={acc_rf:.4f},  F1={f1_rf:.4f}")
    print(f"  XGBoost:             Acc={acc_xgb:.4f},  F1={f1_xgb:.4f}")
    print(f"  SVM (RBF):           Acc={acc_svm:.4f},  F1={f1_svm:.4f}")
    print(f"  Shallow Neural Net:  Acc={acc_mlp_s:.4f},  F1={f1_mlp_s:.4f}")
    print(f"  Deep Neural Net:     Acc={acc_mlp_d:.4f},  F1={f1_mlp_d:.4f}")

def main():
    if not DATA_CSV.exists():
        print(f"Dataset not found at {DATA_CSV}")
        return
        
    df = pd.read_csv(DATA_CSV, low_memory=False)
    
    # Filter for Deadlift
    df_dl = df[df['exercise'] == 'deadlift'].copy()
    print(f"Loaded Deadlift Dataset: {len(df_dl)} frames")
    print(f"  Correct: {(df_dl['form_correct']==True).sum()}")
    print(f"  Incorrect: {(df_dl['form_correct']==False).sum()}")
    
    exclude_cols = ['frame', 'timestamp', 'exercise', 'form_correct', 'risk_level',
                   'deadlift_type', 'video_name', 'score', 'error_type']
    feature_cols = [col for col in df_dl.columns if col not in exclude_cols]
    
    # Ensure there's no string columns by converting to numeric safely
    X = df_dl[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df_dl['form_correct'].astype(int).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
        
    print("\nApplying SMOTE to balance the dataset...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    evaluate_models(X_res, X_test, y_res, y_test)

if __name__ == "__main__":
    main()
