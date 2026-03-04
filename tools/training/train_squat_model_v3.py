# -*- coding: utf-8 -*-
"""
Train Squat Model v3 — The Ultimate Ensemble
Techniques:
  1. Feature Engineering  — angle ratios, depth ratio, asymmetry metrics
  2. SMOTE               — oversample minority class (Correct) to balance
  3. 5-Model Ensemble    — XGBoost + LightGBM + CatBoost + RandomForest + GradientBoosting

Run: python tools/training/train_squat_model_v3.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, f1_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR

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

# ── Feature Engineering ────────────────────────────────────────────────────────
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

# ── Load & Prepare ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(KAGGLE_CSV)
    df = add_engineered_features(df)
    print(f"[OK] Loaded {len(df)} rows, {len(get_feature_cols(df))} features")
    return df

# ── Train Binary Model ─────────────────────────────────────────────────────────
def train_binary(df):
    print("\n" + "="*60)
    print("[TRAIN] Binary Squat Form (Correct/Incorrect) - Ultimate Ensemble")
    print("="*60)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(0).values
    y = (df['label'] != 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nApplying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print("\nTraining XGBoost...")
    m_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                              eval_metric='logloss', random_state=42, n_jobs=-1)
    
    print("Training LightGBM...")
    m_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, 
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    
    print("Training CatBoost...")
    m_cat = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05, 
                               random_seed=42, verbose=0, thread_count=-1)
    
    print("Training RandomForest...")
    m_rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, 
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    
    print("Training GradientBoosting...")
    m_gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                                      subsample=0.8, random_state=42)

    print("\nBuilding 5-Model Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[('xgb', m_xgb), ('lgb', m_lgb), ('cat', m_cat), ('rf', m_rf), ('gb', m_gb)],
        voting='soft'
    )
    ensemble.fit(X_res, y_res)

    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n[OK] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"     F1 macro: {f1:.4f}")
    
    model_path = MODELS_DIR / 'squat_form_v3.pkl'
    joblib.dump({
        'model': ensemble, 'feature_cols': feat_cols, 'model_type': 'binary_ensemble_v3',
        'accuracy': acc, 'f1_macro': f1,
    }, model_path)
    return ensemble, acc, f1

# ── Train Multiclass Model ─────────────────────────────────────────────────────
def train_multiclass(df):
    print("\n" + "="*60)
    print("[TRAIN] Multiclass Squat Error (6 classes) - Ultimate Ensemble")
    print("="*60)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(0).values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nApplying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print("\nTraining XGBoost...")
    m_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                              eval_metric='mlogloss', random_state=42, n_jobs=-1)
    
    print("Training LightGBM...")
    m_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, 
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    
    print("Training CatBoost...")
    m_cat = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05, 
                               loss_function='MultiClass', random_seed=42, verbose=0, thread_count=-1)
    
    print("Training RandomForest...")
    m_rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, 
                                  class_weight='balanced', random_state=42, n_jobs=-1)

    print("\nBuilding 4-Model Voting Ensemble (GB drops multiclass scaling poorly sometimes)...")
    ensemble = VotingClassifier(
        estimators=[('xgb', m_xgb), ('lgb', m_lgb), ('cat', m_cat), ('rf', m_rf)],
        voting='soft'
    )
    ensemble.fit(X_res, y_res)

    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n[OK] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"     F1 macro: {f1:.4f}")
    
    model_path = MODELS_DIR / 'squat_form_detailed_v3.pkl'
    joblib.dump({
        'model': ensemble, 'feature_cols': feat_cols, 'model_type': 'multiclass_ensemble_v3',
        'accuracy': acc, 'f1_macro': f1, 'label_map': LABEL_MAP,
    }, model_path)
    return ensemble, acc, f1

def main():
    df = load_data()
    b_model, b_acc, b_f1 = train_binary(df)
    m_model, m_acc, m_f1 = train_multiclass(df)

if __name__ == "__main__":
    main()
