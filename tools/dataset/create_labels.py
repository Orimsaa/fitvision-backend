# -*- coding: utf-8 -*-
"""
Create Labels from Existing Videos
"""
from pathlib import Path
import json
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

def create_labels():
    print("\nCreating labels from video filenames...")
    
    videos_dir = Path("C:/fit/FitVision/data/raw/videos/deadlift")
    labels_dir = Path("C:/fit/FitVision/data/processed/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping
    label_mapping = {
        'corr': {'form_correct': True, 'error_types': [], 'risk_level': 'low'},
        'oe': {'form_correct': False, 'error_types': ['over_extension'], 'risk_level': 'medium'},
        'rb': {'form_correct': False, 'error_types': ['rounded_back'], 'risk_level': 'high'},
        'ehe': {'form_correct': False, 'error_types': ['excessive_hip_extension'], 'risk_level': 'medium'}
    }
    
    exercise_mapping = {
        'R_': 'romanian',
        'S_': 'sumo',
        'conv_': 'conventional'
    }
    
    videos = list(videos_dir.glob("*.mp4"))
    created = 0
    
    for video_file in videos:
        label_file = labels_dir / f"{video_file.stem}_label.json"
        
        if label_file.exists():
            continue
        
        parts = video_file.stem.split('_')
        
        # Find exercise type
        exercise_type = 'conventional'
        for prefix, ex_type in exercise_mapping.items():
            if video_file.name.startswith(prefix):
                exercise_type = ex_type
                break
        
        # Find error type
        error_key = parts[1] if len(parts) > 1 else 'corr'
        label_info = label_mapping.get(error_key, label_mapping['corr'])
        
        # Create label
        label = {
            "filename": video_file.name,
            "exercise": "deadlift",
            "deadlift_type": exercise_type,
            "form_correct": label_info['form_correct'],
            "error_types": label_info['error_types'],
            "risk_level": label_info['risk_level'],
            "notes": "Auto-labeled from filename"
        }
        
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(label, f, indent=2, ensure_ascii=False)
        created += 1
    
    print(f"Created {created} labels")
    print(f"Total labels: {len(list(labels_dir.glob('*.json')))}")

if __name__ == "__main__":
    create_labels()
