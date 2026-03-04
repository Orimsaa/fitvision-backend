# -*- coding: utf-8 -*-
"""
Generate Full Performance Table across All Exercises
Run: python tools/reporting/generate_full_table.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
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

# Dataset paths
KAGGLE_SQUAT = Path("C:/fit/FitVision/data/raw/kaggle_squat/squat_features_augmented.csv")
TRAINING_CSV = Path("C:/fit/FitVision/data/processed/training_dataset.csv")
BENCHPRESS_CSV = Path("C:/fit/FitVision/data/interim/benchpress_features.csv")

def evaluate_models_for_exercise(X, y, exercise_name):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
        
    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, 
                                                class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                     eval_metric='logloss', use_label_encoder=False, 
                                     random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        "Shallow MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, alpha=0.001,
                                     solver='adam', random_state=42, early_stopping=True),
        "Deep MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, alpha=0.001,
                                  solver='adam', random_state=42, early_stopping=True)
    }
    
    results = []
    
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_res)
        y_pred = model.predict(X_test_scaled)
        
        # Binary Classification Metrics
        acc = accuracy_score(y_test, y_pred)
        # Using binary average since it's Correct(1) vs Incorrect(0)
        # Exception: if labels got flipped, we might just use macro to be safe. We'll use macro.
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            "Exercise": exercise_name.capitalize(),
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
    return results

def get_squat_data():
    df = pd.read_csv(KAGGLE_SQUAT)
    # Target: 0 is correct, != 0 is incorrect. We want 1=Correct, 0=Incorrect 
    # to standardize, let's map it. Output: y=1 means correct
    df['form_correct'] = (df['label'] == 0).astype(int)
    
    # Base features from previous scripts
    base_features = [
        'left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle',
        'left_ankle_angle', 'right_ankle_angle', 'spine_angle', 'torso_lean',
        'left_knee_lateral', 'right_knee_lateral', 'symmetry_score', 'hip_depth'
    ]
    # Engineered
    df['avg_knee_angle'] = (df['left_knee_angle'] + df['right_knee_angle']) / 2
    df['avg_hip_angle']  = (df['left_hip_angle']  + df['right_hip_angle'])  / 2
    df['knee_hip_ratio'] = df['avg_knee_angle'] / (df['avg_hip_angle'] + 1e-8)
    df['knee_depth_ratio'] = df['avg_knee_angle'] / 90.0
    df['ankle_asymmetry'] = (df['left_ankle_angle'] - df['right_ankle_angle']).abs()
    df['hip_asymmetry'] = (df['left_hip_angle'] - df['right_hip_angle']).abs()
    df['total_lateral'] = df['left_knee_lateral'].abs() + df['right_knee_lateral'].abs()
    df['lean_consistency'] = (df['spine_angle'] - df['torso_lean']).abs()
    
    eng_cols = [
        'avg_knee_angle', 'avg_hip_angle', 'knee_hip_ratio',
        'knee_depth_ratio', 'ankle_asymmetry', 'hip_asymmetry',
        'total_lateral', 'lean_consistency'
    ]
    feat_cols = base_features + eng_cols
    
    X = df[feat_cols].fillna(0).values
    y = df['form_correct'].values
    return X, y

def get_deadlift_data():
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    df_dl = df[df['exercise'] == 'deadlift'].copy()
    
    exclude_cols = ['frame', 'timestamp', 'exercise', 'form_correct', 'risk_level',
                   'deadlift_type', 'video_name', 'score', 'error_type']
    feature_cols = [col for col in df_dl.columns if col not in exclude_cols]
    
    X = df_dl[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df_dl['form_correct'].astype(int).values
    return X, y

def get_benchpress_data():
    df = pd.read_csv(BENCHPRESS_CSV, low_memory=False)
    # label == 1 (Correct), 0 (Incorrect)
    
    exclude_cols = ['frame', 'timestamp', 'video_name', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df['label'].astype(int).values
    return X, y

def main():
    print("Gathering Data and Running Massive Evaluation...")
    all_results = []
    
    # 1. Squat
    try:
        print("\n--- Evaluating SQUAT ---")
        X_s, y_s = get_squat_data()
        all_results.extend(evaluate_models_for_exercise(X_s, y_s, "Squat"))
    except Exception as e:
        print(f"Skipping Squat due to error: {e}")
        
    # 2. Deadlift
    try:
        print("\n--- Evaluating DEADLIFT ---")
        X_d, y_d = get_deadlift_data()
        all_results.extend(evaluate_models_for_exercise(X_d, y_d, "Deadlift"))
    except Exception as e:
        print(f"Skipping Deadlift due to error: {e}")
        
    # 3. Bench Press
    try:
        print("\n--- Evaluating BENCH PRESS ---")
        X_b, y_b = get_benchpress_data()
        all_results.extend(evaluate_models_for_exercise(X_b, y_b, "Bench Press"))
    except Exception as e:
        print(f"Skipping Bench Press due to error: {e}")
        
    # Create DataFrame
    res_df = pd.DataFrame(all_results)
    
    # Format percentages
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        res_df[col] = (res_df[col] * 100).map("{:.2f}%".format)
        
    # Save CSV and Markdown
    csv_path = Path("C:/fit/FitVision/data/processed/full_model_report.csv")
    md_path = Path("C:/fit/FitVision/data/processed/full_model_report.md")
    
    res_df.to_csv(csv_path, index=False)
    
    # Generate Markdown Table
    md_content = "# กราฟเปรียบเทียบประสิทธิภาพการวิเคราะห์ท่าทางแบบทวีคูณ (Multi-Exercise Model Performance)\n\n"
    md_content += res_df.to_markdown(index=False)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\nReport Generated at: {md_path}")
    print("\n" + md_content)

if __name__ == "__main__":
    main()
