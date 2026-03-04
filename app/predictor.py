"""
FitVision — Predictor
Loads and runs all exercise models
"""
import numpy as np
import joblib
from pathlib import Path
from functools import lru_cache
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
@lru_cache(maxsize=1)
def get_models():
    models = {}
    paths = {
        "exercise":       MODELS_DIR / "exercise_classifier.pkl",
        "deadlift_form":  MODELS_DIR / "deadlift_form.pkl",
        "squat_binary":   MODELS_DIR / "squat_form.pkl",
        "squat_detailed": MODELS_DIR / "squat_form_detailed.pkl",
        "benchpress_form":MODELS_DIR / "benchpress_form.pkl",
    }
    for name, path in paths.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  [OK] {name}")
        else:
            print(f"  [WARN] Not found: {path}")
    return models


# ── Predict functions ──────────────────────────────────────────────────────────
def predict_exercise(features: list) -> dict:
    models = get_models()
    if "exercise" not in models:
        return {"exercise": "unknown", "confidence": 0}

    m   = models["exercise"]
    X   = np.array(features).reshape(1, -1)
    idx = m["model"].predict(X)[0]
    proba = m["model"].predict_proba(X)[0]

    return {
        "exercise":   EXERCISE_MAP.get(idx, "unknown"),
        "confidence": float(proba[idx]),
    }


def predict_deadlift(features: list) -> dict:
    models = get_models()
    if "deadlift_form" not in models:
        return {"form_correct": True, "confidence": 0, "feedback": "Model not loaded"}

    m     = models["deadlift_form"]
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
    models = get_models()
    if "squat_binary" not in models or "squat_detailed" not in models:
        return {"form_correct": True, "confidence": 0,
                "error_type": "Correct", "feedback": "Model not loaded"}

    feat_vec = engineer_squat_features(squat_features)
    X = np.array(feat_vec).reshape(1, -1)

    # Binary: correct or not
    bm      = models["squat_binary"]
    b_pred  = bm["model"].predict(X)[0]
    b_proba = bm["model"].predict_proba(X)[0]

    # Detailed: which error
    dm      = models["squat_detailed"]
    d_pred  = dm["model"].predict(X)[0]
    d_proba = dm["model"].predict_proba(X)[0]

    form_correct = b_pred == 0
    error_label  = SQUAT_ERROR_MAP.get(d_pred, "Unknown")

    return {
        "form_correct":      form_correct,
        "confidence":        float(b_proba[b_pred]),
        "error_type":        error_label,
        "error_code":        int(d_pred),
        "detail_confidence": float(d_proba[d_pred]),
        "feedback":          "Good squat form! 💪" if form_correct else error_label,
    }


def predict_benchpress(features: list) -> dict:
    models = get_models()
    if "benchpress_form" not in models:
        return {"form_correct": True, "confidence": 0, "feedback": "Model not loaded"}

    m     = models["benchpress_form"]
    X     = np.array(features).reshape(1, -1)
    
    # m["model"] is our VotingClassifier
    pred  = m["model"].predict(X)[0]
    proba = m["model"].predict_proba(X)[0]
    conf  = float(proba[pred])

    # label 0 = Correct, 1 = Incorrect
    form_correct = (pred == 0)

    # General feedback for bench press
    return {
        "form_correct": form_correct,
        "confidence":   conf,
        "feedback":     "Good bench press form! 💪" if form_correct else "Check your form: Keep elbows tucked, back arched, and wrists straight."
    }
