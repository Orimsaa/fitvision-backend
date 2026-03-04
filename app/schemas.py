"""FitVision — Pydantic schemas"""
from pydantic import BaseModel
from typing import Optional

class Features13(BaseModel):
    """13 features for deadlift / exercise classifier"""
    features: list[float]  # exactly 13 values

class SquatFeatures(BaseModel):
    """12 raw squat features — engineered features computed server-side"""
    left_knee_angle:    float
    right_knee_angle:   float
    left_hip_angle:     float
    right_hip_angle:    float
    left_ankle_angle:   float
    right_ankle_angle:  float
    spine_angle:        float
    torso_lean:         float
    left_knee_lateral:  float
    right_knee_lateral: float
    symmetry_score:     float
    hip_depth:          float

class ExercisePrediction(BaseModel):
    exercise:   str
    confidence: float

class FormPrediction(BaseModel):
    form_correct:      bool
    confidence:        float
    feedback:          str
    error_type:        Optional[str] = None
    error_code:        Optional[int] = None
    detail_confidence: Optional[float] = None
