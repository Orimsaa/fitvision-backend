"""
FitVision Configuration Settings
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
YOLO_MODEL_PATH = MODELS_DIR / "yolov8s.pt"  # YOLOv8 for person detection
BENCHPRESS_MODEL_PATH = MODELS_DIR / "benchpress.pkl"
SQUAT_MODEL_PATH = MODELS_DIR / "squat.pkl"
DEADLIFT_MODEL_PATH = MODELS_DIR / "deadlift.pkl"

# MediaPipe settings
MEDIAPIPE_CONFIDENCE = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# Video settings
DEFAULT_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Risk assessment thresholds
RISK_LOW_THRESHOLD = 30
RISK_MEDIUM_THRESHOLD = 60
RISK_HIGH_THRESHOLD = 100

# Exercise types
EXERCISES = {
    0: 'bench_press',
    1: 'squat',
    2: 'deadlift'
}

# Feedback language
FEEDBACK_LANGUAGE = 'th'  # 'th' for Thai, 'en' for English
