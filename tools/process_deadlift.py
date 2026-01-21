# -*- coding: utf-8 -*-
"""
Process Deadlift Videos - Auto-label from filenames
"""
import shutil
from pathlib import Path
import json
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

def process_deadlift_videos():
    print("\n" + "="*60)
    print("Processing Deadlift Videos")
    print("="*60 + "\n")
    
    source_dir = Path("C:/fit/Raw MP4 Videos")
    dest_dir = Path("C:/fit/FitVision/data/raw/videos/deadlift")
    labels_dir = Path("C:/fit/FitVision/data/processed/labels")
    
    dest_dir.mkdir(parents=True, exist_ok=True)
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
    
    copied_count = 0
    labeled_count = 0
    
    # Process all files
    for video_file in source_dir.glob("*.mp4"):
        # Skip flipped files
        if '_flipped' in video_file.name:
            continue
        
        # Copy video
        dest_file = dest_dir / video_file.name
        if not dest_file.exists():
            shutil.copy(video_file, dest_file)
            copied_count += 1
        
        # Create label
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
            "notes": f"Auto-labeled from filename",
            "source": "Raw MP4 Videos"
        }
        
        # Save label
        label_file = labels_dir / f"{video_file.stem}_label.json"
        if not label_file.exists():
            with open(label_file, 'w', encoding='utf-8') as f:
                json.dump(label, f, indent=2, ensure_ascii=False)
            labeled_count += 1
    
    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"   Copied videos: {copied_count} files")
    print(f"   Created labels: {labeled_count} files")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("  1. python tools\\label_data.py (option 2)")
    print("  2. python tools\\train_model.py")

if __name__ == "__main__":
    process_deadlift_videos()
