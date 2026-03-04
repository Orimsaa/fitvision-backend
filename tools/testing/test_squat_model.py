# -*- coding: utf-8 -*-
"""
Test Squat Models
- โหลดโมเดล squat_form.pkl และ squat_form_detailed.pkl
- ทดสอบบน Kaggle data ที่ยังไม่เคย train
- ทดสอบด้วย synthetic poses (correct/incorrect)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR, DATA_DIR

# ── Feature engineering (same as train_squat_model_v2.py) ─────────────────────
def add_engineered_features(df):
    df = df.copy()
    df['avg_knee_angle']    = (df['left_knee_angle']  + df['right_knee_angle']) / 2
    df['avg_hip_angle']     = (df['left_hip_angle']   + df['right_hip_angle'])  / 2
    df['knee_hip_ratio']    = df['avg_knee_angle'] / (df['avg_hip_angle'] + 1e-8)
    df['knee_depth_ratio']  = df['avg_knee_angle'] / 90.0
    df['ankle_asymmetry']   = (df['left_ankle_angle']  - df['right_ankle_angle']).abs()
    df['hip_asymmetry']     = (df['left_hip_angle']    - df['right_hip_angle']).abs()
    df['total_lateral']     = df['left_knee_lateral'].abs() + df['right_knee_lateral'].abs()
    df['lean_consistency']  = (df['spine_angle'] - df['torso_lean']).abs()
    return df

LABEL_MAP = {0: 'Correct', 1: 'Shallow', 2: 'Forward Lean',
             3: 'Knees Caving', 4: 'Heels Off', 5: 'Asymmetric'}


def load_models():
    binary_path  = MODELS_DIR / 'squat_form.pkl'
    detail_path  = MODELS_DIR / 'squat_form_detailed.pkl'

    binary  = joblib.load(binary_path)
    detail  = joblib.load(detail_path)

    print(f"[OK] squat_form.pkl          version={binary.get('model_type','?')}  acc={binary.get('accuracy',0):.4f}")
    print(f"[OK] squat_form_detailed.pkl version={detail.get('model_type','?')}  acc={detail.get('accuracy',0):.4f}")
    return binary, detail


def test_on_kaggle_sample(binary, detail, n_samples=200):
    """ทดสอบบน sample ของ Kaggle data"""
    print("\n" + "="*60)
    print(f"[TEST 1] Kaggle data sample ({n_samples} samples/class)")
    print("="*60)

    csv_path = DATA_DIR / 'raw' / 'kaggle_squat' / 'squat_features_augmented.csv'
    df = pd.read_csv(csv_path)
    df = add_engineered_features(df)

    # Sample equally from each class
    samples = []
    for lbl in range(6):
        s = df[df['label'] == lbl].sample(min(n_samples, len(df[df['label']==lbl])),
                                           random_state=99)
        samples.append(s)
    df_test = pd.concat(samples, ignore_index=True)

    feat_cols = binary['feature_cols']
    X = df_test[feat_cols].fillna(0).values
    y_true_binary = (df_test['label'] != 0).astype(int)
    y_true_multi  = df_test['label'].values

    # Binary predictions
    y_pred_binary = binary['model'].predict(X)
    binary_acc = (y_pred_binary == y_true_binary).mean()
    print(f"\nBinary (correct/incorrect):  Accuracy = {binary_acc:.4f} ({binary_acc*100:.1f}%)")

    # Detailed predictions
    y_pred_multi = detail['model'].predict(X)
    detail_acc = (y_pred_multi == y_true_multi).mean()
    print(f"Detailed (6-class):          Accuracy = {detail_acc:.4f} ({detail_acc*100:.1f}%)")

    # Per-class breakdown
    print("\nDetailed per class:")
    print(f"  {'Class':<15} {'Correct':>8} {'Total':>7} {'Acc':>8}")
    print(f"  {'-'*42}")
    for lbl in range(6):
        mask     = y_true_multi == lbl
        correct  = (y_pred_multi[mask] == lbl).sum()
        total    = mask.sum()
        acc      = correct / total if total > 0 else 0
        bar = '█' * int(acc * 20)
        print(f"  {LABEL_MAP[lbl]:<15} {correct:>8} {total:>7} {acc:>7.1%}  {bar}")


def test_synthetic_poses(binary, detail):
    """ทดสอบด้วย synthetic poses ที่กำหนดเอง"""
    print("\n" + "="*60)
    print("[TEST 2] Synthetic Squat Poses")
    print("="*60)

    # Define pose scenarios
    scenarios = [
        {
            'name': 'Perfect Squat (correct)',
            'data': {
                'left_knee_angle': 92, 'right_knee_angle': 91,
                'left_hip_angle': 95, 'right_hip_angle': 94,
                'left_ankle_angle': 75, 'right_ankle_angle': 76,
                'spine_angle': 15, 'torso_lean': 15,
                'left_knee_lateral': 0.01, 'right_knee_lateral': 0.01,
                'symmetry_score': 3.0, 'hip_depth': 0.65,
            },
            'expected_binary': 0,
            'expected_multi': 0,
        },
        {
            'name': 'Shallow Squat',
            'data': {
                'left_knee_angle': 145, 'right_knee_angle': 143,
                'left_hip_angle': 130, 'right_hip_angle': 130,
                'left_ankle_angle': 82, 'right_ankle_angle': 82,
                'spine_angle': 10, 'torso_lean': 10,
                'left_knee_lateral': 0.01, 'right_knee_lateral': 0.01,
                'symmetry_score': 3.5, 'hip_depth': 0.42,
            },
            'expected_binary': 1,
            'expected_multi': 1,
        },
        {
            'name': 'Knees Caving In',
            'data': {
                'left_knee_angle': 95, 'right_knee_angle': 97,
                'left_hip_angle': 98, 'right_hip_angle': 96,
                'left_ankle_angle': 74, 'right_ankle_angle': 75,
                'spine_angle': 18, 'torso_lean': 18,
                'left_knee_lateral': -0.08, 'right_knee_lateral': -0.09,
                'symmetry_score': 12.0, 'hip_depth': 0.63,
            },
            'expected_binary': 1,
            'expected_multi': 3,
        },
        {
            'name': 'Forward Lean',
            'data': {
                'left_knee_angle': 94, 'right_knee_angle': 93,
                'left_hip_angle': 110, 'right_hip_angle': 109,
                'left_ankle_angle': 68, 'right_ankle_angle': 69,
                'spine_angle': 52, 'torso_lean': 52,
                'left_knee_lateral': 0.02, 'right_knee_lateral': 0.02,
                'symmetry_score': 4.0, 'hip_depth': 0.62,
            },
            'expected_binary': 1,
            'expected_multi': 2,
        },
        {
            'name': 'Heels Off Ground',
            'data': {
                'left_knee_angle': 90, 'right_knee_angle': 91,
                'left_hip_angle': 93, 'right_hip_angle': 94,
                'left_ankle_angle': 115, 'right_ankle_angle': 116,
                'spine_angle': 22, 'torso_lean': 22,
                'left_knee_lateral': 0.01, 'right_knee_lateral': 0.01,
                'symmetry_score': 5.0, 'hip_depth': 0.64,
            },
            'expected_binary': 1,
            'expected_multi': 4,
        },
    ]

    feat_cols = binary['feature_cols']
    correct_binary = 0
    correct_multi  = 0

    print(f"\n  {'Scenario':<28} {'Binary':>8} {'Detailed':>14} {'Status':>8}")
    print(f"  {'-'*62}")

    for s in scenarios:
        row = pd.DataFrame([s['data']])
        row = add_engineered_features(row)
        X   = row[feat_cols].fillna(0).values

        pred_b = binary['model'].predict(X)[0]
        pred_m = detail['model'].predict(X)[0]

        b_label = "Incorrect" if pred_b == 1 else "Correct"
        m_label = LABEL_MAP.get(pred_m, '?')

        b_ok = pred_b == s['expected_binary']
        m_ok = pred_m == s['expected_multi']

        correct_binary += b_ok
        correct_multi  += m_ok

        status = "✓" if (b_ok and m_ok) else ("~" if (b_ok or m_ok) else "✗")
        print(f"  {s['name']:<28} {b_label:>8} {m_label:>14} {status:>8}")

    n = len(scenarios)
    print(f"\n  Binary correct:   {correct_binary}/{n}")
    print(f"  Detailed correct: {correct_multi}/{n}")


def main():
    print("\n" + "="*60)
    print("  Squat Model Test")
    print("="*60)

    binary, detail = load_models()
    test_on_kaggle_sample(binary, detail, n_samples=300)
    test_synthetic_poses(binary, detail)

    print("\n" + "="*60)
    print("[DONE] Test complete")
    print("="*60)


if __name__ == "__main__":
    main()
