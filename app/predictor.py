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

# ── Feature engineering (must match train_squat_model_v2.py) ──────────────────
def engineer_squat_features(f: dict) -> list:
    lk = f["left_knee_angle"];  rk = f["right_knee_angle"]
    lh = f["left_hip_angle"];   rh = f["right_hip_angle"]
    la = f["left_ankle_angle"]; ra = f["right_ankle_angle"]
    sp = f["spine_angle"];      tl = f["torso_lean"]
    ll = f["left_knee_lateral"]; rl = f["right_knee_lateral"]
    sy = f["symmetry_score"];   hd = f["hip_depth"]

    avg_knee       = (lk + rk) / 2
    avg_hip        = (lh + rh) / 2
    knee_hip_ratio = avg_knee / (avg_hip + 1e-8)
    knee_depth     = avg_knee / 90.0
    ankle_asym     = abs(la - ra)
    hip_asym       = abs(lh - rh)
    total_lat      = abs(ll) + abs(rl)
    lean_con       = abs(sp - tl)

    return [lk, rk, lh, rh, la, ra, sp, tl, ll, rl, sy, hd,
            avg_knee, avg_hip, knee_hip_ratio, knee_depth,
            ankle_asym, hip_asym, total_lat, lean_con]


# ── Model loader ───────────────────────────────────────────────────────────────
_current_exercise = None
_active_models = {}

def _force_free_memory():
    """Aggressively free memory back to the OS."""
    gc.collect()
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def _log_memory():
    """Log current memory usage for debugging OOM."""
    try:
        import os
        rss = os.popen('cat /proc/self/status | grep VmRSS').read().strip()
        if rss:
            print(f"[FitVision] 📊 {rss}")
    except Exception:
        pass

def _evict_all_models():
    """Purge all loaded models from RAM."""
    global _current_exercise, _active_models
    for key in list(_active_models.keys()):
        obj = _active_models.pop(key)
        del obj
    _active_models.clear()
    _current_exercise = None
    _force_free_memory()

def _load_single_model(path):
    """Load a single model file, with memory logging."""
    _log_memory()
    print(f"[FitVision] � Loading {path.name}...")
    model = joblib.load(path)
    _log_memory()
    return model

def get_exercise_models(exercise: str):
    """
    Strict Memory Management for Koyeb 512MB RAM Limit.
    NOTE: Squat is handled separately in predict_squat() — it loads models
    one-at-a-time to avoid having both large ensembles in RAM simultaneously.
    """
    global _current_exercise, _active_models
    
    if _current_exercise == exercise and _active_models:
        return _active_models
    
    print(f"\n[FitVision] � RAM Cleanup: Evicting models for '{_current_exercise}'...")
    _evict_all_models()
    
    _current_exercise = exercise
    
    paths = {
        "exercise":       MODELS_DIR / "exercise_classifier.pkl",
        "deadlift_form":  MODELS_DIR / "deadlift_form.pkl",
        "benchpress_form":MODELS_DIR / "benchpress_form.pkl",
    }
    
    try:
        if exercise == "deadlift":
            if paths["deadlift_form"].exists():
                _active_models["deadlift_form"] = _load_single_model(paths["deadlift_form"])
        elif exercise == "benchpress":
            if paths["benchpress_form"].exists():
                _active_models["benchpress_form"] = _load_single_model(paths["benchpress_form"])
        elif exercise == "classifier":
            if paths["exercise"].exists():
                _active_models["exercise"] = _load_single_model(paths["exercise"])
    except MemoryError:
        print(f"[FitVision] ❌ MemoryError loading '{exercise}'!")
        _evict_all_models()
        return {}
        
    print(f"[FitVision] ✅ Model swap complete.")
    return _active_models


# ── Predict functions ──────────────────────────────────────────────────────────
def predict_exercise(features: list) -> dict:
    models = get_exercise_models("classifier")
    m = models.get("exercise")
    if not m:
        return {"exercise": "unknown", "confidence": 0}

    X   = np.array(features).reshape(1, -1)
    idx = m["model"].predict(X)[0]
    proba = m["model"].predict_proba(X)[0]

    return {
        "exercise":   EXERCISE_MAP.get(idx, "unknown"),
        "confidence": float(proba[idx]),
    }


def predict_deadlift(features: list) -> dict:
    models = get_exercise_models("deadlift")
    m = models.get("deadlift_form")
    if not m:
        return {"form_correct": True, "confidence": 0, "feedback": "Model not loaded"}

    X     = np.array(features).reshape(1, -1)
    pred  = m["model"].predict(X)[0]
    proba = m["model"].predict_proba(X)[0]
    conf  = float(proba[pred])

    return {
        "form_correct": bool(pred),
        "confidence":   conf,
        "feedback": "Good form! Keep it up" if pred else "Check your form",
    }


def predict_squat(squat_features: dict) -> dict:
    """
    ⚡ MEMORY-OPTIMIZED SQUAT PREDICTION ⚡
    
    Loads each squat model ONE AT A TIME to avoid OOM on 512MB instances.
    Flow: Load binary → predict → FREE → Load detailed → predict → FREE
    This halves peak RAM vs loading both models simultaneously.
    """
    # First, evict any other models from RAM
    _evict_all_models()
    
    # ── DEBUG: Log raw features from frontend ──
    print(f"\n[FitVision] 🔍 SQUAT DEBUG — Raw features from frontend:")
    for k, v in squat_features.items():
        print(f"  {k:25s} = {v:.4f}")
    
    feat_vec = engineer_squat_features(squat_features)
    X = np.array(feat_vec).reshape(1, -1)
    
    # ── DEBUG: Log engineered features ──
    feat_names = [
        'left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle',
        'left_ankle_angle', 'right_ankle_angle', 'spine_angle', 'torso_lean',
        'left_knee_lateral', 'right_knee_lateral', 'symmetry_score', 'hip_depth',
        'avg_knee', 'avg_hip', 'knee_hip_ratio', 'knee_depth',
        'ankle_asym', 'hip_asym', 'total_lat', 'lean_con'
    ]
    print(f"[FitVision] 🔍 Engineered feature vector ({len(feat_vec)} features):")
    for name, val in zip(feat_names, feat_vec):
        print(f"  {name:25s} = {val:.4f}")
    
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
            print("[FitVision] ✅ Binary model done, freed from RAM")
    except MemoryError:
        print("[FitVision] ❌ MemoryError on squat binary model")
        _force_free_memory()
    
    # ── Stage 2: Multi-class error diagnosis ──
    d_pred = 0
    d_proba = [1.0] + [0.0] * 5
    try:
        if squat_detailed_path.exists():
            dm = _load_single_model(squat_detailed_path)
            d_pred  = dm["model"].predict(X)[0]
            d_proba = dm["model"].predict_proba(X)[0]
            # FREE immediately
            del dm
            _force_free_memory()
            print("[FitVision] ✅ Detailed model done, freed from RAM")
    except MemoryError:
        print("[FitVision] ❌ MemoryError on squat detailed model")
        _force_free_memory()
    
    form_correct = b_pred == 0
    error_label  = SQUAT_ERROR_MAP.get(d_pred, "Unknown")
    
    print(f"[FitVision] 🔍 RESULT: b_pred={b_pred}, b_proba={[f'{p:.3f}' for p in b_proba]}, form_correct={form_correct}")
    print(f"[FitVision] 🔍 RESULT: d_pred={d_pred}, error={error_label}")
    
    _log_memory()
    
    return {
        "form_correct":      form_correct,
        "confidence":        float(b_proba[b_pred]),
        "error_type":        error_label,
        "error_code":        int(d_pred),
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

