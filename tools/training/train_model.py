"""
Model Training Script
Train exercise classification and form analysis models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, MODELS_DIR

class ModelTrainer:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, dataset_path=None):
        """Load training dataset"""
        if dataset_path is None:
            dataset_path = DATA_DIR / 'processed' / 'training_dataset.csv'
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            print(f"[ERROR] Dataset not found: {dataset_path}")
            print("   Run label_data.py first to create dataset")
            return None
        
        df = pd.read_csv(dataset_path, low_memory=False)
        print(f"[OK] Loaded dataset: {len(df)} samples")
        print(f"   Exercises: {df['exercise'].value_counts().to_dict()}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Select feature columns (exclude metadata and string columns)
        exclude_cols = ['frame', 'timestamp', 'exercise', 'form_correct', 'risk_level',
                       'deadlift_type', 'video_name']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        
        print(f"[OK] Features prepared: {X.shape}")
        print(f"   Feature columns: {len(feature_cols)}")
        
        return X, feature_cols
    
    def train_exercise_classifier(self, df):
        """Train exercise type classifier"""
        print("\n" + "="*60)
        print("[TRAIN] Exercise Classifier")
        print("="*60 + "\n")
        
        # Prepare data
        X, feature_cols = self.prepare_features(df)
        
        # Encode labels
        exercise_map = {'benchpress': 0, 'squat': 1, 'deadlift': 2}
        y = df['exercise'].map(exercise_map).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples\n")
        
        # Train model
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[OK] Training completed!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed metrics
        print("\nClassification Report:")
        # Get actual class names from data
        unique_classes = sorted(np.unique(y))
        class_names = [k for k, v in sorted(exercise_map.items(), key=lambda x: x[1]) if v in unique_classes]
        print(classification_report(y_test, y_pred, labels=unique_classes, target_names=class_names))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Save model
        model_path = self.models_dir / 'exercise_classifier.pkl'
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'exercise_map': exercise_map,
            'accuracy': accuracy
        }, model_path)
        
        print(f"\n[SAVED] Model saved: {model_path}")
        
        return model, accuracy
    
    def train_form_analyzer(self, df, exercise_type):
        """Train form correctness classifier for specific exercise"""
        print("\n" + "="*60)
        print(f"[TRAIN] Form Analyzer - {exercise_type.upper()}")
        print("="*60 + "\n")
        
        # Filter by exercise
        df_exercise = df[df['exercise'] == exercise_type].copy()
        
        if len(df_exercise) < 10:
            print(f"[ERROR] Not enough data for {exercise_type}: {len(df_exercise)} samples")
            print("   Need at least 10 samples")
            return None, 0
        
        print(f"Dataset: {len(df_exercise)} samples")
        print(f"  Correct: {(df_exercise['form_correct']==True).sum()}")
        print(f"  Incorrect: {(df_exercise['form_correct']==False).sum()}\n")
        
        # Prepare data
        X, feature_cols = self.prepare_features(df_exercise)
        y = df_exercise['form_correct'].astype(int).values
        
        # Check if we have both classes
        if len(np.unique(y)) < 2:
            print(f"[ERROR] Need both correct and incorrect examples!")
            return None, 0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples\n")
        
        # Train model
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[OK] Training completed!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Incorrect', 'Correct']))
        
        # Save model
        model_path = self.models_dir / f'{exercise_type}_form.pkl'
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'accuracy': accuracy
        }, model_path)
        
        print(f"\n[SAVED] Model saved: {model_path}")
        
        return model, accuracy
    
    def train_all(self, dataset_path=None):
        """Train all models"""
        # Load dataset
        df = self.load_dataset(dataset_path)
        
        if df is None:
            return
        
        results = {}
        
        # Train exercise classifier
        model, acc = self.train_exercise_classifier(df)
        results['exercise_classifier'] = acc
        
        # Train form analyzers for each exercise
        for exercise in ['benchpress', 'squat', 'deadlift']:
            model, acc = self.train_form_analyzer(df, exercise)
            if model is not None:
                results[f'{exercise}_form'] = acc
        
        # Summary
        print("\n" + "="*60)
        print("[SUMMARY] TRAINING SUMMARY")
        print("="*60)
        
        for model_name, accuracy in results.items():
            print(f"  {model_name:30s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n[OK] All models trained successfully!")
        print(f"   Models saved in: {self.models_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train exercise analysis models')
    parser.add_argument('--dataset', type=str, help='Path to training dataset CSV')
    parser.add_argument('--exercise', type=str, choices=['benchpress', 'squat', 'deadlift'],
                       help='Train only specific exercise form analyzer')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    if args.exercise:
        # Train specific exercise form analyzer
        df = trainer.load_dataset(args.dataset)
        if df is not None:
            trainer.train_form_analyzer(df, args.exercise)
    else:
        # Train all models
        trainer.train_all(args.dataset)

if __name__ == "__main__":
    main()
