"""
FitVision — Predictor
Loads and runs all exercise models
"""
import numpy as np
import joblib
from pathlib import Path
import gc
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import MODELS_DIR

# ── Label maps ─────────────────────────────────────────────────────────────────
SQUAT_ERROR_MAP = {
    0: "Correct",
    1: "Shallow squat — ลงให้ลึกกว่านี้",
    2: "Forward lean — อย่าโน้มตัวไปข้างหน้า",
    3: "Knees caving in — เก็บเข่าให้ตรง",
    4: "Heels off ground — ส้นเท้าต้องติดพื้น",
    5: "Asymmetric — ทั้งสองข้างไม่สมมาตร",
}

EXERCISE_MAP = {0: "benchpress", 1: "squat", 2: "deadlift"}

def engineer_squat_features(f: dict) -> list:
    """Full 20-feature extraction matching the v7 powerlifting training script."""
    lk = f.get("left_knee_angle", 180);  rk = f.get("right_knee_angle", 180)
    lh = f.get("left_hip_angle", 180);   rh = f.get("right_hip_angle", 180)
    la = f.get("left_ankle_angle", 160); ra = f.get("right_ankle_angle", 160)
    sp = f.get("spine_angle", 0);        tl = f.get("torso_lean", 0)
    ll = f.get("left_knee_lateral", 0);  rl = f.get("right_knee_lateral", 0)
    sy = f.get("symmetry_score", 0);     hd = f.get("hip_depth", 0.5)

    avg_knee       = (lk + rk) / 2
    avg_hip        = (lh + rh) / 2
    knee_hip_ratio = avg_knee / (avg_hip + 1e-8)
    knee_depth     = avg_knee / 90.0
    ankle_asym     = abs(la - ra)
    hip_asym       = abs(lh - rh)
    total_lat      = abs(ll) + abs(rl)
    lean_con       = abs(sp - tl)

    # MUST match get_feature_cols() in train_powerlifting_ml.py exact order
    return [lk, rk, lh, rh, la, ra, sp, tl, ll, rl, sy, hd,
            avg_knee, avg_hip, knee_hip_ratio, knee_depth,
            ankle_asym, hip_asym, total_lat, lean_con]

def predict_squat(squat_features: dict) -> dict:
    """
    ⚡ MEMORY-OPTIMIZED ML SQUAT PREDICTION ⚡
    
    Loads each squat model ONE AT A TIME to avoid OOM on 512MB instances.
    Uses the new perfectly-matched v7 models trained on Powerlifting raw landmarks.
    """
    # First, evict any other models from RAM
    _evict_all_models()
    
    feat_vec = engineer_squat_features(squat_features)
    X = np.array(feat_vec).reshape(1, -1)
    
    # Check if they are just standing (avg_knee > 150)
    # If so, return correct immediately without loading models to save time
    if feat_vec[12] > 150:
        return {
            "form_correct": True,
            "confidence": 0.85,
            "error_type": "Correct",
            "error_code": 0,
            "detail_confidence": 0.85,
            "feedback": "Good form — squat deeper to begin analysis 💪",
        }
    
    squat_binary_path   = MODELS_DIR / "squat_form.pkl"
    squat_detailed_path = MODELS_DIR / "squat_form_detailed.pkl"
    
    # ── Stage 1: Binary classification (Correct / Incorrect) ──
    b_pred = 0
    b_proba = [1.0, 0.0]
    try:
        if squat_binary_path.exists():
            bm = _load_single_model(squat_binary_path)
            b_pred  = bm["model"].predict(X)[0]
            b_proba = bm["model"].predict_proba(X)[0]
            # FREE immediately
            del bm
            _force_free_memory()
            print("[FitVision] ✅ Binary model done")
    except MemoryError:
        print("[FitVision] ❌ MemoryError on squat binary model")
        _force_free_memory()
    
    form_correct = (b_pred == 0)
    
    # ── Stage 2: Multi-class error diagnosis ──
    # Only load the multiclass model if the form was incorrect
    d_pred = 0
    d_proba = [1.0] + [0.0] * 5
    
    if not form_correct:
        try:
            if squat_detailed_path.exists():
                dm = _load_single_model(squat_detailed_path)
                d_pred  = dm["model"].predict(X)[0]
                d_proba = dm["model"].predict_proba(X)[0]
                # FREE immediately
                del dm
                _force_free_memory()
                print("[FitVision] ✅ Detailed model done")
            else:
                d_pred = 1 # Fallback to shallow
        except MemoryError:
            print("[FitVision] ❌ MemoryError on squat detailed model")
            _force_free_memory()
            d_pred = 1
    
    error_label  = SQUAT_ERROR_MAP.get(d_pred, "Unknown error")
    
    print(f"[FitVision] 🔍 ML SQUAT: b_pred={b_pred} d_pred={d_pred} error={error_label}")
    
    return {
        "form_correct":      form_correct,
        "confidence":        float(b_proba[b_pred]),
        "error_type":        "Correct" if form_correct else error_label,
        "error_code":        int(d_pred) if not form_correct else 0,
        "detail_confidence": float(d_proba[d_pred]),
        "feedback":          "Good squat form! 💪" if form_correct else error_label,
    }


def predict_benchpress(features: list) -> dict:
    models = get_exercise_models("benchpress")
    m = models.get("benchpress_form")
    if not m:
        return {"form_correct": True, "confidence": 0, "feedback": "Model not loaded"}

    X     = np.array(features).reshape(1, -1)
    
    pred  = m["model"].predict(X)[0]
    proba = m["model"].predict_proba(X)[0]
    conf  = float(proba[pred])

    form_correct = (pred == 0)

    return {
        "form_correct": form_correct,
        "confidence":   conf,
        "feedback":     "Good bench press form! 💪" if form_correct else "Check your form: Keep elbows tucked, back arched, and wrists straight."
    }

