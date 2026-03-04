# -*- coding: utf-8 -*-
"""
Create Training Dataset - Combine features and labels
"""
import pandas as pd
import json
from pathlib import Path
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

def create_training_dataset():
    print("\n" + "="*60)
    print("Creating Training Dataset")
    print("="*60 + "\n")
    
    features_dir = Path("C:/fit/FitVision/data/processed/features")
    labels_dir = Path("C:/fit/FitVision/data/processed/labels")
    output_file = Path("C:/fit/FitVision/data/processed/training_dataset.csv")
    
    all_data = []
    
    # Process each feature file
    feature_files = list(features_dir.glob("*_features.csv"))
    print(f"Found {len(feature_files)} feature files")
    
    processed = 0
    skipped = 0
    
    for feature_file in feature_files:
        # Find corresponding label
        video_name = feature_file.stem.replace('_features', '')
        label_file = labels_dir / f"{video_name}_label.json"
        
        if not label_file.exists():
            skipped += 1
            continue
        
        # Load label
        with open(label_file, 'r', encoding='utf-8') as f:
            label = json.load(f)
        
        # Load features
        df = pd.read_csv(feature_file)
        
        # Add label info to each frame
        df['exercise'] = label['exercise']
        df['deadlift_type'] = label.get('deadlift_type', 'conventional')
        df['form_correct'] = label['form_correct']
        df['risk_level'] = label['risk_level']
        df['video_name'] = video_name
        
        all_data.append(df)
        processed += 1
        
        if processed % 50 == 0:
            print(f"Processed: {processed}/{len(feature_files)}")
    
    if not all_data:
        print("[ERROR] No data to combine!")
        return
    
    # Combine all data
    print("\nCombining data...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"   Processed: {processed} videos")
    print(f"   Skipped: {skipped} videos")
    print(f"   Total frames: {len(final_df)}")
    print(f"   Output: {output_file}")
    print(f"{'='*60}\n")
    
    # Show statistics
    print("Dataset Statistics:")
    print(f"   Deadlift types:")
    print(final_df.groupby('deadlift_type')['video_name'].nunique())
    print(f"\n   Form correctness:")
    print(final_df.groupby('form_correct')['video_name'].nunique())
    print(f"\n   Risk levels:")
    print(final_df.groupby('risk_level')['video_name'].nunique())

if __name__ == "__main__":
    create_training_dataset()
