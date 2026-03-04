import sys
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from tools.features.extract_features import FeatureExtractor

# Windows UTF-8 console fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

def process_benchpress_videos():
    print("\n" + "="*60)
    print("1. Copying Bench Press Videos")
    print("="*60 + "\n")

    src_correct = Path(r"C:\Users\tonkla\Downloads\Correct-20260301T193101Z-1-001\Correct")
    src_incorrect = Path(r"C:\Users\tonkla\Downloads\Incorrect-20260301T193119Z-1-001\Incorrect")
    
    # Destination directories
    raw_dir = Path(r"C:\fit\FitVision\data\raw\videos\benchpress")
    dest_correct = raw_dir / "correct"
    dest_incorrect = raw_dir / "incorrect"
    
    dest_correct.mkdir(parents=True, exist_ok=True)
    dest_incorrect.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_files(src, dest, prefix):
        files = list(src.glob("*.[mM][pP]4")) + list(src.glob("*.[mM][oO][vV]"))
        copied = 0
        for f in tqdm(files, desc=f"Copying {prefix}"):
            dest_file = dest / f.name
            if not dest_file.exists():
                shutil.copy2(f, dest_file)
            copied += 1
        return copied

    n_corr = copy_files(src_correct, dest_correct, "Correct")
    n_inc = copy_files(src_incorrect, dest_incorrect, "Incorrect")
    print(f"\nCopied {n_corr} Correct and {n_inc} Incorrect videos to {raw_dir}")

    print("\n" + "="*60)
    print("2. Extracting Bench Press Features")
    print("="*60 + "\n")

    out_dir = Path(r"C:\fit\FitVision\data\interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "benchpress_features.csv"

    extractor = FeatureExtractor()
    all_data = []

    def extract_folder(folder_path, label, class_name):
        files = list(folder_path.glob("*.[mM][pP]4")) + list(folder_path.glob("*.[mM][oO][vV]"))
        print(f"\nProcessing {len(files)} {class_name} videos...")
        for i, video_path in enumerate(files):
            print(f"\n[{class_name}] {i+1}/{len(files)}: {video_path.name}")
            try:
                df = extractor.extract_from_video(video_path)
                if df is not None and not df.empty:
                    df['video_name'] = video_path.name
                    df['label'] = label
                    all_data.append(df)
            except Exception as e:
                print(f"  [ERROR] Failed to process {video_path.name}: {e}")

    extract_folder(dest_correct, 0, "Correct")
    extract_folder(dest_incorrect, 1, "Incorrect")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(out_csv, index=False)
        print("\n" + "="*60)
        print(f"Extraction Complete. Saved {len(final_df)} frames to {out_csv}")
        print("="*60 + "\n")
    else:
        print("\n[ERROR] No data extracted from any videos.")

if __name__ == "__main__":
    process_benchpress_videos()
