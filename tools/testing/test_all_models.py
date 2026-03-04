import cv2
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.features.extract_features import FeatureExtractor
from app.predictor import get_models, predict_exercise, predict_squat, predict_deadlift, predict_benchpress

def main():
    print("\n" + "="*60)
    print("FITVISION MULTI-MODEL LIVE TESTING")
    print("="*60)
    
    # 1. Warm-up Models
    print("\nLoading models... (Exercise, Squat, Deadlift, Bench Press)")
    get_models()

    # 2. Get Video Input
    print("\nChoose input method:")
    print("1. Webcam (0)")
    print("2. Test Video (Provide path)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        cap_source = 0
    else:
        video_path = input("Enter video path: ").strip()
        cap_source = video_path

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {cap_source}")
        sys.exit(1)

    extractor = FeatureExtractor()

    print("\n[INFO] Press 'q' to quit window.")

    # Keeping a rolling average of exercise classification to avoid flickering
    exercise_history = []
    HISTORY_LEN = 15

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break
        
        display_frame = frame.copy()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = extractor.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw landmarks
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            mp_drawing.draw_landmarks(
                display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract features dict
            features_dict = extractor.extract_features_from_landmarks(results.pose_landmarks)
            
            # 13 Base features for Exercise / Deadlift / BenchPress
            base_cols = ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
                         'right_shoulder_angle', 'left_hip_angle', 'right_hip_angle', 
                         'left_knee_angle', 'right_knee_angle', 'shoulder_width', 
                         'hip_width', 'torso_length', 'elbow_symmetry', 'knee_symmetry']
            
            base_array = [features_dict.get(k, 0.0) for k in base_cols]
            
            # 1. Predict Exercise Type
            ex_res = predict_exercise(base_array)
            detected_exercise = ex_res['exercise']
            ex_conf = ex_res['confidence']
            
            # Smooth out the detection
            exercise_history.append(detected_exercise)
            if len(exercise_history) > HISTORY_LEN:
                exercise_history.pop(0)
            
            # Majority vote for current exercise
            current_exercise = max(set(exercise_history), key=exercise_history.count)
            
            # Draw Exercise Classification
            cv2.putText(display_frame, f"Exe: {current_exercise.upper()} ({ex_conf*100:.0f}%)", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            # 2. Predict Form based on current exercise
            form_res = None
            if current_exercise == 'squat':
                form_res = predict_squat(features_dict)
            elif current_exercise == 'deadlift':
                form_res = predict_deadlift(base_array)
            elif current_exercise == 'benchpress':
                form_res = predict_benchpress(base_array)

            # 3. Display Form Results
            if form_res:
                is_correct = form_res['form_correct']
                conf = form_res['confidence']
                feedback = form_res.get('feedback', '')
                
                label = "GOOD FORM" if is_correct else "INCORRECT"
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                
                cv2.putText(display_frame, f"Form: {label} ({conf*100:.0f}%)", 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                if not is_correct:
                    # Provide specific feedback
                    cv2.putText(display_frame, f"Fix: {feedback}", 
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No Pose Detected", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("FitVision Multi-Model Analysis", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
