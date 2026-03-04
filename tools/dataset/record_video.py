"""
Video Recording Tool for Data Collection
Record exercise videos with webcam and save with metadata
"""
import cv2
import json
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

class VideoRecorder:
    def __init__(self, exercise_type, person_name):
        self.exercise_type = exercise_type  # 'benchpress', 'squat', 'deadlift'
        self.person_name = person_name
        self.output_dir = DATA_DIR / 'raw' / 'videos' / exercise_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def record(self):
        """Record video from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{self.exercise_type}_{self.person_name}_{timestamp}.mp4"
        video_path = self.output_dir / video_filename
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        print(f"\n{'='*60}")
        print(f"🎥 Recording: {video_filename}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("  - Press SPACE to start/stop recording")
        print("  - Press 'q' to quit without saving")
        print("  - Press 's' to save and quit")
        print(f"\nExercise: {self.exercise_type.upper()}")
        print(f"Person: {self.person_name}")
        print(f"\n{'='*60}\n")
        
        recording = False
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # Display status
            if recording:
                cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display_frame, "RECORDING", (50, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Frames: {frame_count}", (50, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                out.write(frame)
                frame_count += 1
            else:
                cv2.putText(display_frame, "READY - Press SPACE to record", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"{self.exercise_type.upper()}", (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Person: {self.person_name}", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('Video Recorder', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - toggle recording
                recording = not recording
                if recording:
                    print("🔴 Recording started...")
                else:
                    print("⏸️  Recording paused")
            
            elif key == ord('s'):  # Save and quit
                if frame_count > 0:
                    print(f"\n✅ Video saved: {video_path}")
                    print(f"   Frames: {frame_count}")
                    print(f"   Duration: {frame_count/fps:.1f}s")
                    
                    # Save metadata
                    metadata = {
                        'filename': video_filename,
                        'exercise': self.exercise_type,
                        'person': self.person_name,
                        'timestamp': timestamp,
                        'frames': frame_count,
                        'fps': fps,
                        'duration_seconds': frame_count / fps,
                        'resolution': f"{width}x{height}"
                    }
                    
                    metadata_path = video_path.with_suffix('.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    break
                else:
                    print("⚠️  No frames recorded!")
            
            elif key == ord('q'):  # Quit without saving
                print("\n❌ Recording cancelled")
                video_path.unlink(missing_ok=True)
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return video_path if frame_count > 0 else None

def main():
    print("\n" + "="*60)
    print("🎥 FitVision - Video Recording Tool")
    print("="*60 + "\n")
    
    # Get input
    print("Exercise types: benchpress, squat, deadlift")
    exercise = input("Enter exercise type: ").strip().lower()
    
    if exercise not in ['benchpress', 'squat', 'deadlift']:
        print("❌ Invalid exercise type!")
        return
    
    person = input("Enter person name: ").strip()
    
    if not person:
        print("❌ Person name required!")
        return
    
    # Record
    recorder = VideoRecorder(exercise, person)
    video_path = recorder.record()
    
    if video_path:
        print(f"\n✅ Success! Video saved to:")
        print(f"   {video_path}")
        
        # Ask if want to record another
        another = input("\nRecord another video? (y/n): ").strip().lower()
        if another == 'y':
            main()
    else:
        print("\n❌ No video saved")

if __name__ == "__main__":
    main()
