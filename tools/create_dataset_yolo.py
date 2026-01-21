"""
Create training dataset from YOLO features and labels

Combines features from features_yolo/ with labels from labels_yolo/
to create a complete training dataset.

Usage:
    python tools/create_dataset_yolo.py
"""

import pandas as pd
import json
from pathlib import Path

def main():
    """Main function"""
    
    # Paths
    features_dir = Path("data/processed/features_yolo")
    labels_dir = Path("data/processed/labels_yolo")
    output_file = Path("data/processed/training_dataset_yolo.csv")
    
    # Get all feature files
    feature_files = sorted(features_dir.glob("*_features_yolo.csv"))
    
    print(f"\nFound {len(feature_files)} feature files")
    print("="*60)
    
    all_data = []
    processed = 0
    skipped = 0
    
    for feature_file in feature_files:
        # Get corresponding label file
        video_name = feature_file.stem.replace('_features_yolo', '')
        label_file = labels_dir / f"{video_name}_label.json"
        
        if not label_file.exists():
            print(f"Warning: No label for {feature_file.name}")
            skipped += 1
            continue
        
        # Load features
        try:
            features_df = pd.read_csv(feature_file)
        except Exception as e:
            print(f"Error loading {feature_file.name}: {e}")
            skipped += 1
            continue
        
        # Load label
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label = json.load(f)
        except Exception as e:
            print(f"Error loading {label_file.name}: {e}")
            skipped += 1
            continue
        
        # Add label columns to features
        features_df['exercise'] = label['exercise']
        features_df['deadlift_type'] = label['deadlift_type']
        features_df['form_correct'] = label['form_correct']
        features_df['risk_level'] = label['risk_level']
        features_df['video_name'] = video_name
        
        all_data.append(features_df)
        processed += 1
        
        if processed % 50 == 0:
            print(f"Processed {processed} videos...")
    
    # Combine all data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Save
        final_df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print(f"Dataset created successfully!")
        print(f"Videos processed: {processed}")
        print(f"Videos skipped: {skipped}")
        print(f"Total samples: {len(final_df)}")
        print(f"Saved to: {output_file}")
        
        # Show distribution
        print("\n" + "="*60)
        print("Dataset Distribution:")
        print(f"\nForm Correct:")
        print(final_df['form_correct'].value_counts())
        print(f"\nDeadlift Type:")
        print(final_df['deadlift_type'].value_counts())
    else:
        print("\nError: No data to save!")


if __name__ == "__main__":
    main()
