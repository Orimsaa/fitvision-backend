# -*- coding: utf-8 -*-
"""
Train Bench Press Form Model
Labels:
  0 = Correct
  1 = Incorrect (General form errors based on the provided dataset)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR

CSV_PATH = Path(r"C:\fit\FitVision\data\interim\benchpress_features.csv")

# Same 12 features used for squat + elbows for bench press.
# Wait, extract_features.py returns 13 features for deadlift/exercise,
# and specifically for the base FeatureExtractor it returns:
# elbow_angle, shoulder_angle, hip_angle, knee_angle (left/right) -> 8
# widths/lengths -> 3
# symmetry -> 2
# Total: 13 features.
# Let's inspect columns dynamically based on what was saved.

def load_data():
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV not found at {CSV_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    
    # Drop metadata columns to find strictly numerical features
    exclude_cols = ['frame', 'timestamp', 'video_name', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"[OK] Loaded {len(df)} rows.")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    return df, feature_cols

def train_binary_model(df, feature_cols):
    print("\n" + "="*60)
    print("[TRAIN] Binary Bench Press Form (Correct vs Incorrect)")
    print("="*60)
    # Prepare data
    groups = df['video_name'].values
    X = df[feature_cols].fillna(0).values
    y = df['label'].values  # 0 for correct, 1 for incorrect

    # Use GroupShuffleSplit to prevent data leakage from same videos
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain shape: {X_train.shape}   Test shape: {X_test.shape}")
    print(f"Class Distribution -> Correct (0): {(y_train==0).sum()}, Incorrect (1): {(y_train==1).sum()}")

    # Apply SMOTE to balance classes in training
    print("\nApplying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE -> Correct (0): {(y_res==0).sum()}, Incorrect (1): {(y_res==1).sum()}")

    # 1. XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )
    
    # 2. Random Forest
    print("Training RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=20,
        class_weight='balanced',
        random_state=42, 
        n_jobs=1
    )

    # Ensemble
    print("Building Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model)],
        voting='soft'
    )
    
    ensemble.fit(X_res, y_res)

    # Evaluate
    print("\nEvaluating Model on Test Set...")
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n[RESULTS] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"[RESULTS] F1 Macro: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Correct', 'Incorrect']))

    # Save Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / 'benchpress_form.pkl'
    
    joblib.dump({
        'model': ensemble,
        'feature_cols': feature_cols,
        'model_type': 'binary_ensemble',
        'accuracy': acc,
        'f1_macro': f1,
    }, model_path)
    
    print(f"\n[SAVED] Model successfully saved to: {model_path}")
    return acc, f1

def main():
    df, feature_cols = load_data()
    train_binary_model(df, feature_cols)

if __name__ == "__main__":
    main()
