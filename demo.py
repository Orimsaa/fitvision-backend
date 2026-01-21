# -*- coding: utf-8 -*-
"""
Test Trained Models - Simple Demo
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import MODELS_DIR

class DeadliftAnalyzer:
    def __init__(self):
        # Load models
        print("Loading models...")
        self.exercise_model = joblib.load(MODELS_DIR / 'exercise_classifier.pkl')
        self.form_model = joblib.load(MODELS_DIR / 'deadlift_form.pkl')
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        print("[OK] Models loaded!")
    
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
    
    def extract_features(self, landmarks):
        """Extract features from landmarks"""
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
        
        features = [
            left_elbow_angle,
            right_elbow_angle,
            self.calculate_angle(left_elbow, left_shoulder, left_hip),
            self.calculate_angle(right_elbow, right_shoulder, right_hip),
            self.calculate_angle(left_shoulder, left_hip, left_knee),
            self.calculate_angle(right_shoulder, right_hip, right_knee),
            left_knee_angle,
            right_knee_angle,
            np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder)),
            np.linalg.norm(np.array(left_hip) - np.array(right_hip)),
            np.linalg.norm(np.array(left_shoulder) - np.array(left_hip)),
            abs(left_elbow_angle - right_elbow_angle),
            abs(left_knee_angle - right_knee_angle),
        ]
        
        return np.array(features).reshape(1, -1)
    
    def analyze_frame(self, frame):
        """Analyze a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, frame
        
        # Extract features
        features = self.extract_features(results.pose_landmarks)
        
        # Predict form
        form_pred = self.form_model['model'].predict(features)[0]
        form_proba = self.form_model['model'].predict_proba(features)[0]
        
        # Draw skeleton
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Prepare result
        result = {
            'form_correct': bool(form_pred),
            'confidence': float(form_proba[form_pred]),
            'incorrect_prob': float(form_proba[0]),
            'correct_prob': float(form_proba[1])
        }
        
        # Draw result on frame
        h, w = frame.shape[:2]
        
        if result['form_correct']:
            color = (0, 255, 0)  # Green
            text = "CORRECT FORM"
        else:
            color = (0, 0, 255)  # Red
            text = "INCORRECT FORM"
        
        cv2.rectangle(frame, (10, 10), (w-10, 100), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"Confidence: {result['confidence']*100:.1f}%", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result, frame
    
    def analyze_webcam(self):
        """Analyze from webcam"""
        print("\nStarting webcam...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result, annotated_frame = self.analyze_frame(frame)
            
            cv2.imshow('Deadlift Form Analyzer', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def analyze_video(self, video_path):
        """Analyze a video file"""
        print(f"\nAnalyzing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        correct_count = 0
        incorrect_count = 0
        total_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result, annotated_frame = self.analyze_frame(frame)
            
            if result:
                total_frames += 1
                if result['form_correct']:
                    correct_count += 1
                else:
                    incorrect_count += 1
            
            cv2.imshow('Deadlift Form Analyzer', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total frames analyzed: {total_frames}")
        print(f"Correct form: {correct_count} ({correct_count/total_frames*100:.1f}%)")
        print(f"Incorrect form: {incorrect_count} ({incorrect_count/total_frames*100:.1f}%)")
        print("="*60)

def main():
    analyzer = DeadliftAnalyzer()
    
    print("\n" + "="*60)
    print("Deadlift Form Analyzer")
    print("="*60)
    print("\nOptions:")
    print("  1. Analyze from webcam")
    print("  2. Analyze video file")
    print("  3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        analyzer.analyze_webcam()
    elif choice == '2':
        video_path = input("Enter video path: ").strip()
        if Path(video_path).exists():
            analyzer.analyze_video(video_path)
        else:
            print(f"[ERROR] Video not found: {video_path}")
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()
