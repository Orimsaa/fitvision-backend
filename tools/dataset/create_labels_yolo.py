"""
Create labels from video filenames for YOLO features

This script automatically generates labels based on filename patterns:
- R_corr_* → Romanian, Correct
- R_oe_* → Romanian, Over Extension
- S_ehe_* → Sumo, Excessive Hip Extension
etc.

Usage:
    python tools/create_labels_yolo.py
"""

import json
from pathlib import Path

def create_label_from_filename(filename):
    """
    Create label from filename pattern
    
    Format: {type}_{error}_{number}.mp4
    - type: R (Romanian), S (Sumo), conv (Conventional)
    - error: corr, oe, rb, ehe
    """
    # Remove extension and _features_yolo suffix
    name = filename.replace('_features_yolo.csv', '')
    parts = name.split('_')
    
    # Deadlift type mapping
    type_map = {
        'R': 'romanian',
        'S': 'sumo',
        'conv': 'conventional'
    }
    
    if parts[0] in type_map:
        deadlift_type = type_map[parts[0]]
    else:
        print(f"Warning: Unknown type in {filename}")
        return None
    
    # Error type mapping
    error_map = {
        'corr': {
            'form_correct': True,
            'error_types': [],
            'risk_level': 'low'
        },
        'oe': {
            'form_correct': False,
            'error_types': ['over_extension'],
            'risk_level': 'medium'
        },
        'rb': {
            'form_correct': False,
            'error_types': ['rounded_back'],
            'risk_level': 'high'
        },
        'ehe': {
            'form_correct': False,
            'error_types': ['excessive_hip_extension'],
            'risk_level': 'medium'
        }
    }
    
    # Get error key
    if len(parts) >= 2:
        error_key = parts[1]
        if error_key in error_map:
            error_info = error_map[error_key]
        else:
            print(f"Warning: Unknown error type in {filename}")
            return None
    else:
        print(f"Warning: Invalid filename format {filename}")
        return None
    
    return {
        'filename': name,
        'exercise': 'deadlift',
        'deadlift_type': deadlift_type,
        'form_correct': error_info['form_correct'],
        'error_types': error_info['error_types'],
        'risk_level': error_info['risk_level']
    }


def main():
    """Main function"""
    
    # Paths
    features_dir = Path("data/processed/features_yolo")
    labels_dir = Path("data/processed/labels_yolo")
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all feature files
    feature_files = list(features_dir.glob("*_features_yolo.csv"))
    
    print(f"\nFound {len(feature_files)} feature files")
    print("="*60)
    
    created = 0
    skipped = 0
    
    for feature_file in feature_files:
        # Create label
        label = create_label_from_filename(feature_file.name)
        
        if label is None:
            skipped += 1
            continue
        
        # Save label
        label_file = labels_dir / f"{label['filename']}_label.json"
        
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(label, f, indent=2, ensure_ascii=False)
        
        created += 1
        
        if created % 50 == 0:
            print(f"Created {created} labels...")
    
    print("\n" + "="*60)
    print(f"Labels created: {created}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {labels_dir}")


if __name__ == "__main__":
    main()
