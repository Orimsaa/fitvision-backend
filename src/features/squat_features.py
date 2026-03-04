# -*- coding: utf-8 -*-
"""
Squat-Specific Feature Extractor
Extracts squat-relevant features from MediaPipe Pose landmarks:
  - Knee angles (left/right)
  - Hip angles (left/right)
  - Ankle angles (left/right)
  - Spine angle
  - Torso lean
  - Lateral knee deviation (left/right)
  - Symmetry score
  - Hip depth

These match the Kaggle 'Squat Exercise Pose Dataset' columns directly.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_DIR


class SquatFeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def angle_3pts(self, a, b, c):
        """Angle at point b, formed by a-b-c"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def extract_squat_features(self, landmarks):
        """Extract squat-specific features matching Kaggle dataset columns"""
        lm = landmarks.landmark
        P = self.mp_pose.PoseLandmark

        def pt(landmark_id):
            l = lm[landmark_id.value]
            return [l.x, l.y]

        # Key landmarks
        l_shoulder = pt(P.LEFT_SHOULDER)
        r_shoulder = pt(P.RIGHT_SHOULDER)
        l_hip      = pt(P.LEFT_HIP)
        r_hip      = pt(P.RIGHT_HIP)
        l_knee     = pt(P.LEFT_KNEE)
        r_knee     = pt(P.RIGHT_KNEE)
        l_ankle    = pt(P.LEFT_ANKLE)
        r_ankle    = pt(P.RIGHT_ANKLE)
        l_foot     = pt(P.LEFT_FOOT_INDEX)
        r_foot     = pt(P.RIGHT_FOOT_INDEX)

        mid_hip      = [(l_hip[0]+r_hip[0])/2,      (l_hip[1]+r_hip[1])/2]
        mid_shoulder = [(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2]

        # Knee angles: hip - knee - ankle
        left_knee_angle  = self.angle_3pts(l_hip, l_knee, l_ankle)
        right_knee_angle = self.angle_3pts(r_hip, r_knee, r_ankle)

        # Hip angles: shoulder - hip - knee
        left_hip_angle  = self.angle_3pts(l_shoulder, l_hip, l_knee)
        right_hip_angle = self.angle_3pts(r_shoulder, r_hip, r_knee)

        # Ankle angles: knee - ankle - foot
        left_ankle_angle  = self.angle_3pts(l_knee, l_ankle, l_foot)
        right_ankle_angle = self.angle_3pts(r_knee, r_ankle, r_foot)

        # Spine angle: angle of mid_hip → mid_shoulder relative to vertical
        # (0° = perfectly upright, larger = more leaned)
        vertical    = [mid_hip[0], mid_hip[1] - 1.0]  # point directly above hip
        spine_angle = self.angle_3pts(vertical, mid_hip, mid_shoulder)

        # Torso lean: same as spine_angle (degree of forward lean)
        torso_lean = spine_angle

        # Lateral knee deviation: horizontal offset of knee vs ankle (normalised)
        left_knee_lateral  = l_knee[0] - l_ankle[0]   # negative = caving in (for left)
        right_knee_lateral = r_ankle[0] - r_knee[0]   # negative = caving in (for right)

        # Symmetry score: sum of left/right angle differences
        symmetry_score = abs(left_knee_angle - right_knee_angle) + \
                         abs(left_hip_angle  - right_hip_angle)

        # Hip depth: y-coordinate of mid-hip (larger y = lower = deeper squat)
        hip_depth = mid_hip[1]

        return {
            'left_knee_angle':    left_knee_angle,
            'right_knee_angle':   right_knee_angle,
            'left_hip_angle':     left_hip_angle,
            'right_hip_angle':    right_hip_angle,
            'left_ankle_angle':   left_ankle_angle,
            'right_ankle_angle':  right_ankle_angle,
            'spine_angle':        spine_angle,
            'torso_lean':         torso_lean,
            'left_knee_lateral':  left_knee_lateral,
            'right_knee_lateral': right_knee_lateral,
            'symmetry_score':     symmetry_score,
            'hip_depth':          hip_depth,
        }

    def extract_from_video(self, video_path):
        """Extract squat features from a video file"""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {video_path.name}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        all_features = []
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.name}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            if results.pose_landmarks:
                feat = self.extract_squat_features(results.pose_landmarks)
                feat['frame']     = frame_idx
                feat['timestamp'] = frame_idx / fps
                all_features.append(feat)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        if not all_features:
            print(f"[ERROR] No pose detected in {video_path.name}")
            return None

        return pd.DataFrame(all_features)
