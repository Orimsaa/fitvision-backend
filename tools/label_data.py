"""
Data Labeling Tool
Simple GUI for labeling extracted features
"""
import pandas as pd
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

class DataLabeler:
    def __init__(self):
        self.features_dir = DATA_DIR / 'processed' / 'features'
        self.labels_dir = DATA_DIR / 'processed' / 'labels'
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
    def list_unlabeled_files(self):
        """List all feature files that haven't been labeled"""
        if not self.features_dir.exists():
            return []
        
        feature_files = list(self.features_dir.glob('*_features.csv'))
        unlabeled = []
        
        for f in feature_files:
            label_file = self.labels_dir / f"{f.stem.replace('_features', '')}_label.json"
            if not label_file.exists():
                unlabeled.append(f)
        
        return unlabeled
    
    def label_file(self, feature_file):
        """Label a single feature file"""
        feature_file = Path(feature_file)
        
        # Load features
        df = pd.read_csv(feature_file)
        
        print(f"\n{'='*60}")
        print(f"📝 Labeling: {feature_file.name}")
        print(f"{'='*60}")
        print(f"\nFeatures shape: {df.shape}")
        print(f"Duration: {df['timestamp'].max():.1f}s")
        print(f"Frames: {len(df)}")
        
        # Get labels
        print(f"\n{'='*60}")
        print("LABELING")
        print(f"{'='*60}\n")
        
        # Exercise type
        print("Exercise type:")
        print("  1. Bench Press")
        print("  2. Squat")
        print("  3. Deadlift")
        exercise_choice = input("Enter choice (1-3): ").strip()
        
        exercise_map = {'1': 'benchpress', '2': 'squat', '3': 'deadlift'}
        exercise = exercise_map.get(exercise_choice, 'unknown')
        
        # Form quality
        print("\nForm quality:")
        print("  1. Correct (good form)")
        print("  2. Incorrect (has errors)")
        form_choice = input("Enter choice (1-2): ").strip()
        
        form_correct = form_choice == '1'
        
        # Error types (if incorrect)
        error_types = []
        if not form_correct:
            print("\nError types (comma-separated):")
            print("  - knee_valgus")
            print("  - rounded_back")
            print("  - excessive_arch")
            print("  - asymmetric")
            print("  - shallow_depth")
            print("  - bar_path_deviation")
            print("  - other")
            errors_input = input("Enter errors: ").strip()
            if errors_input:
                error_types = [e.strip() for e in errors_input.split(',')]
        
        # Risk level
        print("\nRisk level:")
        print("  1. Low (0-30)")
        print("  2. Medium (31-60)")
        print("  3. High (61-100)")
        risk_choice = input("Enter choice (1-3): ").strip()
        
        risk_map = {'1': 'low', '2': 'medium', '3': 'high'}
        risk_level = risk_map.get(risk_choice, 'low')
        
        # Additional notes
        notes = input("\nAdditional notes (optional): ").strip()
        
        # Create label
        label = {
            'filename': feature_file.name,
            'exercise': exercise,
            'form_correct': form_correct,
            'error_types': error_types,
            'risk_level': risk_level,
            'notes': notes,
            'num_frames': len(df),
            'duration_seconds': float(df['timestamp'].max())
        }
        
        # Save label
        label_file = self.labels_dir / f"{feature_file.stem.replace('_features', '')}_label.json"
        with open(label_file, 'w') as f:
            json.dump(label, f, indent=2)
        
        print(f"\n✅ Label saved: {label_file.name}")
        
        return label
    
    def create_dataset(self):
        """Create training dataset from labeled features"""
        print("\n📊 Creating dataset from labeled features...")
        
        # Find all labeled files
        label_files = list(self.labels_dir.glob('*_label.json'))
        
        if not label_files:
            print("❌ No labeled files found!")
            return None
        
        print(f"Found {len(label_files)} labeled files")
        
        all_data = []
        
        for label_file in label_files:
            # Load label
            with open(label_file, 'r') as f:
                label = json.load(f)
            
            # Find corresponding features
            feature_file = self.features_dir / label['filename']
            
            if not feature_file.exists():
                print(f"⚠️  Features not found: {feature_file.name}")
                continue
            
            # Load features
            df = pd.read_csv(feature_file)
            
            # Add labels to each frame
            df['exercise'] = label['exercise']
            df['form_correct'] = label['form_correct']
            df['risk_level'] = label['risk_level']
            
            all_data.append(df)
        
        if not all_data:
            print("❌ No data to combine!")
            return None
        
        # Combine all data
        dataset = pd.concat(all_data, ignore_index=True)
        
        # Save dataset
        dataset_path = DATA_DIR / 'processed' / 'training_dataset.csv'
        dataset.to_csv(dataset_path, index=False)
        
        print(f"\n✅ Dataset created: {dataset_path}")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Exercises: {dataset['exercise'].value_counts().to_dict()}")
        print(f"   Form correct: {dataset['form_correct'].value_counts().to_dict()}")
        
        return dataset_path

def main():
    labeler = DataLabeler()
    
    print("\n" + "="*60)
    print("📝 FitVision - Data Labeling Tool")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("  1. Label unlabeled files")
        print("  2. Create training dataset")
        print("  3. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == '1':
            unlabeled = labeler.list_unlabeled_files()
            
            if not unlabeled:
                print("\n✅ All files are labeled!")
                continue
            
            print(f"\nFound {len(unlabeled)} unlabeled files:")
            for i, f in enumerate(unlabeled, 1):
                print(f"  {i}. {f.name}")
            
            file_choice = input(f"\nChoose file to label (1-{len(unlabeled)}) or 'all': ").strip()
            
            if file_choice.lower() == 'all':
                for f in unlabeled:
                    labeler.label_file(f)
                    print()
            else:
                try:
                    idx = int(file_choice) - 1
                    if 0 <= idx < len(unlabeled):
                        labeler.label_file(unlabeled[idx])
                except ValueError:
                    print("❌ Invalid choice!")
        
        elif choice == '2':
            labeler.create_dataset()
        
        elif choice == '3':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
