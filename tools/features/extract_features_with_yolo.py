"""
Extract features from videos with YOLOv8 person detection and ROI cropping

This script:
1. Uses YOLOv8 to detect person in each frame
2. Crops ROI (Region of Interest) with padding
3. Uses MediaPipe Pose on the cropped ROI
4. Extracts 13 features
5. Saves to CSV

Usage:
    python tools/extract_features_with_yolo.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import json

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize YOLOv8
print("Loading YOLOv8 model...")
yolo_model = YOLO('yolov8s.pt')
print("YOLOv8 loaded successfully!")


def detect_person(frame, yolo_model):
    """
    Detect person using YOLOv8
    
    Returns:
        dict: Person info with bounding box, confidence, area
        None: If no person detected
    """
    results = yolo_model(frame, verbose=False)
    
    persons = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        
        if cls == 0:  # Person class
            x_center, y_center, width, height = box.xywh[0].cpu().numpy()
            confidence = float(box.conf[0])
            area = width * height
            
            persons.append({
                'box': (x_center, y_center, width, height),
                'conf': confidence,
                'area': area
            })
    
    if not persons:
        return None
    
    # Select largest person
    largest_person = max(persons, key=lambda p: p['area'])
    return largest_person


def crop_roi_with_padding(frame, person_info, padding_percent=0.15):
    """
    Crop ROI from frame with padding
    
    Args:
        frame: Input frame
        person_info: Person detection info from YOLOv8
        padding_percent: Padding percentage (default 15%)
    
    Returns:
        roi: Cropped region
        coords: (x1, y1, x2, y2) coordinates
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Get bounding box
    x_center, y_center, width, height = person_info['box']
    
    # Convert to corner coordinates
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Calculate padding
    padding_x = width * padding_percent
    padding_y = height * padding_percent
    
    # Apply padding
    x1_padded = x1 - padding_x
    y1_padded = y1 - padding_y
    x2_padded = x2 + padding_x
    y2_padded = y2 + padding_y
    
    # Clip to frame boundaries
    x1_final = max(0, int(x1_padded))
    y1_final = max(0, int(y1_padded))
    x2_final = min(frame_width, int(x2_padded))
    y2_final = min(frame_height, int(y2_padded))
    
    # Crop ROI
    roi = frame[y1_final:y2_final, x1_final:x2_final]
    
    return roi, (x1_final, y1_final, x2_final, y2_final)


def calculate_angle(a, b, c):
    """Calculate angle at point b"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def extract_features(landmarks):
    """Extract 13 features from pose landmarks"""
    
    # Get key landmarks
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    
    features = []
    
    # 8 joint angles
    features.append(calculate_angle(left_shoulder, left_elbow, left_wrist))
    features.append(calculate_angle(right_shoulder, right_elbow, right_wrist))
    features.append(calculate_angle(left_elbow, left_shoulder, left_hip))
    features.append(calculate_angle(right_elbow, right_shoulder, right_hip))
    features.append(calculate_angle(left_shoulder, left_hip, left_knee))
    features.append(calculate_angle(right_shoulder, right_hip, right_knee))
    features.append(calculate_angle(left_hip, left_knee, left_ankle))
    features.append(calculate_angle(right_hip, right_knee, right_ankle))
    
    # 3 distances
    shoulder_width = np.linalg.norm(
        np.array([left_shoulder.x, left_shoulder.y]) -
        np.array([right_shoulder.x, right_shoulder.y])
    )
    features.append(shoulder_width)
    
    hip_width = np.linalg.norm(
        np.array([left_hip.x, left_hip.y]) -
        np.array([right_hip.x, right_hip.y])
    )
    features.append(hip_width)
    
    torso_length = np.linalg.norm(
        np.array([left_shoulder.x, left_shoulder.y]) -
        np.array([left_hip.x, left_hip.y])
    )
    features.append(torso_length)
    
    # 2 symmetry measures
    elbow_symmetry = abs(features[0] - features[1])
    features.append(elbow_symmetry)
    
    knee_symmetry = abs(features[6] - features[7])
    features.append(knee_symmetry)
    
    return features


def process_video_with_yolo(video_path, output_dir):
    """
    Process video with YOLOv8 + MediaPipe pipeline
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save features
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_features = []
    frame_count = 0
    detected_count = 0
    
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Step 1: YOLOv8 detect person
            person_info = detect_person(frame, yolo_model)
            
            if person_info is None:
                continue
            
            # Step 2: Crop ROI with padding
            roi, coords = crop_roi_with_padding(frame, person_info)
            
            if roi.size == 0:
                continue
            
            # Step 3: MediaPipe pose estimation on ROI
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_roi)
            
            if not results.pose_landmarks:
                continue
            
            # Step 4: Extract features
            features = extract_features(results.pose_landmarks.landmark)
            
            all_features.append({
                'frame': frame_count,
                'person_confidence': person_info['conf'],
                'roi_coords': coords,
                **{f'feature_{i+1}': f for i, f in enumerate(features)}
            })
            
            detected_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"(Detected: {detected_count})")
    
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving progress...")
        print(f"Processed {frame_count}/{total_frames} frames so far")
    
    finally:
        cap.release()
        
        # Save to CSV (even if interrupted)
        if all_features:
            df = pd.DataFrame(all_features)
            output_file = output_dir / f"{video_path.stem}_features_yolo.csv"
            df.to_csv(output_file, index=False)
            
            print(f"\nSaved {len(all_features)} frames to {output_file}")
            print(f"Detection rate: {detected_count}/{total_frames} "
                  f"({detected_count/total_frames*100:.1f}%)")
        else:
            print(f"\nWarning: No features extracted from {video_path.name}")


def main():
    """Main function"""
    
    # Get script directory
    script_dir = Path(__file__).parent.parent
    
    # Paths (absolute)
    video_dir = script_dir / "data" / "raw" / "videos" / "deadlift"
    output_dir = script_dir / "data" / "processed" / "features_yolo"
    
    # Check if video directory exists
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        print(f"Please check the path and try again.")
        return
    
    # Get all videos
    video_files = list(video_dir.glob("*.mp4"))
    
    print(f"\nVideo directory: {video_dir}")
    print(f"Found {len(video_files)} videos")
    print("="*60)
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        process_video_with_yolo(video_path, output_dir)
    
    print("\n" + "="*60)
    print("All videos processed!")
    print(f"Features saved to: {output_dir}")


if __name__ == "__main__":
    main()
