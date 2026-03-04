import cv2
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.features.extract_features import FeatureExtractor
from app.predictor import predict_benchpress, get_models

def test_video(video_path, extractor, expected_correct):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0
    
    total_frames = 0
    correct_frames = 0
    
    feat_cols = ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
                 'right_shoulder_angle', 'left_hip_angle', 'right_hip_angle', 
                 'left_knee_angle', 'right_knee_angle', 'shoulder_width', 
                 'hip_width', 'torso_length', 'elbow_symmetry', 'knee_symmetry']
                 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = extractor.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            features_dict = extractor.extract_features_from_landmarks(results.pose_landmarks)
            feature_array = [features_dict.get(k, 0.0) for k in feat_cols]
            
            pred = predict_benchpress(feature_array)
            total_frames += 1
            if pred['form_correct']:
                correct_frames += 1
                
    cap.release()
    
    if total_frames == 0:
        return None, 0, 0
        
    accuracy = correct_frames / total_frames
    
    # We say the video is predicted "Correct" if > 50% of frames are Correct
    predicted_correct = accuracy > 0.5
    
    is_prediction_matched = (predicted_correct == expected_correct)
    
    return is_prediction_matched, accuracy, total_frames

def main():
    print("Loading models...")
    get_models()
    extractor = FeatureExtractor()
    
    base_dir = Path(r"C:\fit\FitVision\data\raw\videos\benchpress")
    correct_dir = base_dir / "correct"
    incorrect_dir = base_dir / "incorrect"
    
    correct_videos = list(correct_dir.glob("*.mp4")) + list(correct_dir.glob("*.mov"))
    incorrect_videos = list(incorrect_dir.glob("*.mp4")) + list(incorrect_dir.glob("*.mov"))
    
    # Test a sample of them (e.g., 10 from each category to be fast)
    test_correct = correct_videos[:10]
    test_incorrect = incorrect_videos[:10]
    
    print(f"\nEvaluating {len(test_correct)} Correct Videos...")
    correct_matched = 0
    for v in tqdm(test_correct, desc="Correct Videos"):
        matched, acc, frames = test_video(str(v), extractor, expected_correct=True)
        if matched is not None and matched:
            correct_matched += 1
            
    print(f"\nEvaluating {len(test_incorrect)} Incorrect Videos...")
    incorrect_matched = 0
    for v in tqdm(test_incorrect, desc="Incorrect Videos"):
        matched, acc, frames = test_video(str(v), extractor, expected_correct=False)
        if matched is not None and matched:
            incorrect_matched += 1
            
    print("\n" + "="*45)
    print("BATCH TEST RESULTS (Video-Level Accuracy)")
    print("="*45)
    print(f"Correct Videos Match:   {correct_matched} / {len(test_correct)} videos")
    print(f"Incorrect Videos Match: {incorrect_matched} / {len(test_incorrect)} videos")
    print("="*45)

if __name__ == '__main__':
    main()
