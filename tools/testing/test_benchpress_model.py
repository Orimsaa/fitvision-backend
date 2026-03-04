import cv2
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.features.extract_features import FeatureExtractor
from app.predictor import predict_benchpress, get_models

def main():
    print("\n" + "="*60)
    print("Testing Bench Press Model (Real-Time)")
    print("="*60)
    
    # 1. Warm-up Models
    print("\nLoading models...")
    get_models()

    # 2. Get Video Input
    print("\nChoose input method:")
    print("1. Webcam (0)")
    print("2. Test Video (e.g. data/raw/videos/benchpress/correct/corr_01.mp4)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        cap_source = 0
    else:
        # Fallback to a default video to test it easily
        video_path = r"C:\fit\FitVision\data\raw\videos\benchpress\correct\corr_01.mp4"
        custom_path = input(f"Enter video path or press Enter to use default [{video_path}]: ").strip()
        if custom_path:
            video_path = custom_path
        cap_source = video_path

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {cap_source}")
        sys.exit(1)

    extractor = FeatureExtractor()

    print("\n[INFO] Press 'q' to quit window.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break
        
        # Keep original frame for drawing
        display_frame = frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = extractor.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw landmarks
            extractor.mp_pose.Pose(
                static_image_mode=False, model_complexity=1
            ) # Just for the drawing utility...
            
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            mp_drawing.draw_landmarks(
                display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract features and predict
            features_dict = extractor.extract_features_from_landmarks(results.pose_landmarks)
            
            # Keep order of features based on training phase
            feat_cols = ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
                         'right_shoulder_angle', 'left_hip_angle', 'right_hip_angle', 
                         'left_knee_angle', 'right_knee_angle', 'shoulder_width', 
                         'hip_width', 'torso_length', 'elbow_symmetry', 'knee_symmetry']
            
            # Build an array
            feature_array = [features_dict.get(k, 0.0) for k in feat_cols]
            
            # Predict
            pred_result = predict_benchpress(feature_array)
            
            # Display results
            label = "CORRECT" if pred_result['form_correct'] else "INCORRECT"
            color = (0, 255, 0) if pred_result['form_correct'] else (0, 0, 255)
            
            cv2.putText(display_frame, f"Bench Press: {label}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            cv2.putText(display_frame, f"Conf: {pred_result['confidence']*100:.1f}%", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
            # Show specific feedback if incorrect
            if not pred_result['form_correct']:
                cv2.putText(display_frame, pred_result['feedback'], (20, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No Pose Detected", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show Output
        cv2.imshow("Bench Press Analysis", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
