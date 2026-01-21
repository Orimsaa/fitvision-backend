"""
Train model with YOLO features and compare with original

This script:
1. Trains Random Forest with YOLO features
2. Evaluates performance
3. Compares with original model
4. Saves comparison report

Usage:
    python tools/compare_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
import json

def prepare_features(df):
    """Prepare features for training"""
    
    # Check which column format we have
    if 'left_elbow_angle' in df.columns:
        # Original format
        feature_columns = [
            'left_elbow_angle', 'right_elbow_angle',
            'left_shoulder_angle', 'right_shoulder_angle',
            'left_hip_angle', 'right_hip_angle',
            'left_knee_angle', 'right_knee_angle',
            'shoulder_width', 'hip_width', 'torso_length',
            'elbow_symmetry', 'knee_symmetry'
        ]
    else:
        # YOLO format (feature_1, feature_2, ...)
        feature_columns = [f'feature_{i}' for i in range(1, 14)]
    
    return df[feature_columns].values


def train_and_evaluate(dataset_path, model_name):
    """Train model and return metrics"""
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print('='*60)
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples")
    
    # Prepare features and labels
    X = prepare_features(df)
    y = df['form_correct'].astype(int)
    
    print(f"\nDataset distribution:")
    print(f"Correct:   {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"Incorrect: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=['Incorrect', 'Correct'],
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results: {model_name}")
    print('='*60)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Incorrect', 'Correct']
    ))
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Incorrect  Correct")
    print(f"Actual Incorrect  {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"       Correct    {cm[1][0]:6d}    {cm[1][1]:6d}")
    
    # Return metrics
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_incorrect': report['Incorrect']['precision'],
        'recall_incorrect': report['Incorrect']['recall'],
        'f1_incorrect': report['Incorrect']['f1-score'],
        'precision_correct': report['Correct']['precision'],
        'recall_correct': report['Correct']['recall'],
        'f1_correct': report['Correct']['f1-score'],
        'confusion_matrix': cm.tolist(),
        'samples_train': len(X_train),
        'samples_test': len(X_test),
        'model': model
    }


def compare_models(original_metrics, yolo_metrics):
    """Compare two models and print results"""
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print('='*60)
    
    # Create comparison table
    print(f"\n{'Metric':<25} {'Original':<15} {'YOLO':<15} {'Difference':<15}")
    print('-'*70)
    
    metrics = [
        ('Accuracy', 'accuracy', '%'),
        ('Precision (Incorrect)', 'precision_incorrect', '%'),
        ('Recall (Incorrect)', 'recall_incorrect', '%'),
        ('F1-Score (Incorrect)', 'f1_incorrect', '%'),
        ('Precision (Correct)', 'precision_correct', '%'),
        ('Recall (Correct)', 'recall_correct', '%'),
        ('F1-Score (Correct)', 'f1_correct', '%'),
    ]
    
    for name, key, unit in metrics:
        orig = original_metrics[key] * 100
        yolo = yolo_metrics[key] * 100
        diff = yolo - orig
        
        print(f"{name:<25} {orig:>6.2f}{unit:<8} {yolo:>6.2f}{unit:<8} {diff:>+6.2f}{unit:<8}")
    
    # Determine winner
    print(f"\n{'='*60}")
    if yolo_metrics['accuracy'] > original_metrics['accuracy']:
        improvement = (yolo_metrics['accuracy'] - original_metrics['accuracy']) * 100
        print(f"[+] YOLO model is BETTER by {improvement:.2f}%")
    elif yolo_metrics['accuracy'] < original_metrics['accuracy']:
        decline = (original_metrics['accuracy'] - yolo_metrics['accuracy']) * 100
        print(f"[-] Original model is BETTER by {decline:.2f}%")
    else:
        print(f"[=] Both models have EQUAL accuracy")
    print('='*60)


def main():
    """Main function"""
    
    # Paths
    original_dataset = Path("data/processed/training_dataset.csv")
    yolo_dataset = Path("data/processed/training_dataset_yolo.csv")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train original model
    print("\n" + "="*60)
    print("STEP 1: Training Original Model (MediaPipe only)")
    print("="*60)
    original_metrics = train_and_evaluate(original_dataset, "Original (MediaPipe)")
    
    # Save original model
    original_model_path = models_dir / "deadlift_form_original.pkl"
    joblib.dump(original_metrics['model'], original_model_path)
    print(f"\nSaved to: {original_model_path}")
    
    # Train YOLO model
    print("\n" + "="*60)
    print("STEP 2: Training YOLO Model (YOLOv8 + ROI + MediaPipe)")
    print("="*60)
    yolo_metrics = train_and_evaluate(yolo_dataset, "YOLO (YOLOv8 + ROI)")
    
    # Save YOLO model
    yolo_model_path = models_dir / "deadlift_form_yolo.pkl"
    joblib.dump(yolo_metrics['model'], yolo_model_path)
    print(f"\nSaved to: {yolo_model_path}")
    
    # Compare models
    compare_models(original_metrics, yolo_metrics)
    
    # Save comparison report
    comparison = {
        'original': {k: v for k, v in original_metrics.items() if k != 'model'},
        'yolo': {k: v for k, v in yolo_metrics.items() if k != 'model'},
        'winner': 'yolo' if yolo_metrics['accuracy'] > original_metrics['accuracy'] else 'original'
    }
    
    report_path = models_dir / "model_comparison.json"
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison report saved to: {report_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Original Model: {original_metrics['accuracy']*100:.2f}% accuracy")
    print(f"YOLO Model:     {yolo_metrics['accuracy']*100:.2f}% accuracy")
    print(f"\nBoth models saved to: {models_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
