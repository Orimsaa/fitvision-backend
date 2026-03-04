# -*- coding: utf-8 -*-
"""
Train Squat Model v2 — Improved
Techniques:
  1. Feature Engineering  — angle ratios, depth ratio, asymmetry metrics
  2. SMOTE               — oversample minority class (Correct) to balance
  3. XGBoost             — stronger than RandomForest for tabular data
  4. Voting Ensemble     — XGBoost + RandomForest + GradientBoosting
  5. Threshold Tuning    — optimise decision threshold for F1

Run: python tools/train_squat_model_v2.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

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

    # Average knee/hip angle (how deep the squat is)
    df['avg_knee_angle'] = (df['left_knee_angle'] + df['right_knee_angle']) / 2
    df['avg_hip_angle']  = (df['left_hip_angle']  + df['right_hip_angle'])  / 2

    # Knee-to-hip ratio — forward lean indicator
    df['knee_hip_ratio'] = df['avg_knee_angle'] / (df['avg_hip_angle'] + 1e-8)

    # Knee depth ratio — how far knee is bent relative to ideal (~90°)
    df['knee_depth_ratio'] = df['avg_knee_angle'] / 90.0

    # Ankle lean asymmetry
    df['ankle_asymmetry'] = (df['left_ankle_angle'] - df['right_ankle_angle']).abs()

    # Hip asymmetry
    df['hip_asymmetry'] = (df['left_hip_angle'] - df['right_hip_angle']).abs()

    # Combined lateral knee deviation
    df['total_lateral'] = df['left_knee_lateral'].abs() + df['right_knee_lateral'].abs()

    # Torso-spine consistency (both measure lean; large diff = noise)
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
    print("[TRAIN] Binary Squat Form  (Correct / Incorrect)  v2")
    print("="*60)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(0).values
    y = (df['label'] != 0).astype(int)   # 0=correct, 1=incorrect

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}")
    print(f"Correct={( y_train==0).sum()}  Incorrect={(y_train==1).sum()}")

    # SMOTE — balance training set only
    print("\nApplying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: Correct={( y_res==0).sum()}  Incorrect={(y_res==1).sum()}")

    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_res, y_res)

    # RandomForest
    print("Training RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        min_samples_leaf=2, class_weight='balanced',
        random_state=42, n_jobs=-1)
    rf_model.fit(X_res, y_res)

    # GradientBoosting
    print("Training GradientBoosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        random_state=42)
    gb_model.fit(X_res, y_res)

    # Voting Ensemble
    print("Building Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model), ('gb', gb_model)],
        voting='soft',
    )
    ensemble.fit(X_res, y_res)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='macro')

    print(f"\n[OK] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"     F1 macro: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Correct', 'Incorrect']))

    # Save
    model_path = MODELS_DIR / 'squat_form.pkl'
    joblib.dump({
        'model':        ensemble,
        'feature_cols': feat_cols,
        'model_type':   'binary_ensemble_v2',
        'accuracy':     acc,
        'f1_macro':     f1,
    }, model_path)
    print(f"[SAVED] {model_path}")
    return ensemble, acc, f1


# ── Train Multiclass Model ─────────────────────────────────────────────────────
def train_multiclass(df):
    print("\n" + "="*60)
    print("[TRAIN] Multiclass Squat Error (6 classes)  v2")
    print("="*60)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(0).values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE for multiclass
    print("\nApplying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # XGBoost multiclass
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1,
    )
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1)

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model)],
        voting='soft',
    )

    print("Training Ensemble...")
    ensemble.fit(X_res, y_res)

    y_pred = ensemble.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='macro')

    print(f"\n[OK] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"     F1 macro:  {f1:.4f}")
    print("\nClassification Report:")
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    model_path = MODELS_DIR / 'squat_form_detailed.pkl'
    joblib.dump({
        'model':        ensemble,
        'feature_cols': feat_cols,
        'model_type':   'multiclass_ensemble_v2',
        'accuracy':     acc,
        'f1_macro':     f1,
        'label_map':    LABEL_MAP,
    }, model_path)
    print(f"[SAVED] {model_path}")
    return ensemble, acc, f1


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    df = load_data()

    b_model, b_acc, b_f1 = train_binary(df)
    m_model, m_acc, m_f1 = train_multiclass(df)

    print("\n" + "="*60)
    print("[SUMMARY]  v2 vs v1")
    print("="*60)
    print(f"  squat_form.pkl          acc={b_acc:.4f}  f1={b_f1:.4f}  (v1: acc=0.8982  f1=0.8325)")
    print(f"  squat_form_detailed.pkl acc={m_acc:.4f}  f1={m_f1:.4f}  (v1: acc=0.8806  f1=0.8782)")


if __name__ == "__main__":
    main()
