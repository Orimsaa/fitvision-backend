# -*- coding: utf-8 -*-
"""
Feature Extraction - Fixed for Thai filenames
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Fix Windows console encoding for Thai characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

class FeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def extract_features_from_landmarks(self, landmarks):
        """Extract features from pose landmarks"""
        lm = landmarks.landmark
        
        # Key points
        left_shoulder = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_elbow = [lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_wrist = [lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_hip = [lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_ankle = [lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Calculate angles
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        features = {
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            'left_shoulder_angle': self.calculate_angle(left_elbow, left_shoulder, left_hip),
            'right_shoulder_angle': self.calculate_angle(right_elbow, right_shoulder, right_hip),
            'left_hip_angle': self.calculate_angle(left_shoulder, left_hip, left_knee),
            'right_hip_angle': self.calculate_angle(right_shoulder, right_hip, right_knee),
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'shoulder_width': np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder)),
            'hip_width': np.linalg.norm(np.array(left_hip) - np.array(right_hip)),
            'torso_length': np.linalg.norm(np.array(left_shoulder) - np.array(left_hip)),
            'elbow_symmetry': abs(left_elbow_angle - right_elbow_angle),
            'knee_symmetry': abs(left_knee_angle - right_knee_angle),
        }
        
        return features
    
    def extract_from_video(self, video_path):
        """Extract features from a video file"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"[ERROR] Video not found")
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n[VIDEO] Processing: {video_path.name}")
        print(f"   Frames: {total_frames}, FPS: {fps:.1f}")
        
        all_features = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Extracting")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                features = self.extract_features_from_landmarks(results.pose_landmarks)
                features['frame'] = frame_idx
                features['timestamp'] = frame_idx / fps
                all_features.append(features)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if not all_features:
            print("[ERROR] No pose detected")
            return None
        
        df = pd.DataFrame(all_features)
        print(f"[OK] Extracted {len(df)} frames ({len(df)/total_frames*100:.1f}%)")
        
        return df
    
    def save_features(self, df, output_path):
        """Save features to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SAVED] {output_path.name}")

def process_all_videos():
    """Process all videos"""
    video_dir = DATA_DIR / 'raw' / 'videos'
    output_dir = DATA_DIR / 'processed' / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.rglob('*.mp4'))
    
    print(f"\n{'='*60}")
    print(f"Found {len(video_files)} videos")
    print(f"{'='*60}\n")
    
    extractor = FeatureExtractor()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        
        output_path = output_dir / f"{video_path.stem}_features.csv"
        
        # Skip if already processed
        if output_path.exists():
            print(f"[SKIP] Already processed: {video_path.name}")
            continue
        
        df = extractor.extract_from_video(video_path)
        
        if df is not None:
            extractor.save_features(df, output_path)

if __name__ == "__main__":
    process_all_videos()
