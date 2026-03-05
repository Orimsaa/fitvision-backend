"""
FitVision — FastAPI main app
Run: uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.schemas import Features13, SquatFeatures, ExercisePrediction, FormPrediction
from app.predictor import predict_exercise, predict_deadlift, predict_squat, predict_benchpress

app = FastAPI(title="FitVision API", version="2.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static web files ────────────────────────────────────────────────────
WEB_DIR = Path(__file__).parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("\n[FitVision] API Starting up. Models will be lazy-loaded on demand to save memory.")
    print("[FitVision] Ready!\n")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "FitVision API running", "docs": "/docs"}

@app.get("/health")
async def health():
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / "data" / "models"
    model_files = [f.name for f in models_dir.glob("*.pkl")] if models_dir.exists() else []
    return {
        "status": "ok",
        "models": model_files,
    }


@app.post("/predict/exercise", response_model=ExercisePrediction)
async def predict_exercise_endpoint(body: Features13):
    if len(body.features) != 13:
        raise HTTPException(400, f"Expected 13 features, got {len(body.features)}")
    return predict_exercise(body.features)

@app.post("/predict/deadlift", response_model=FormPrediction)
async def predict_deadlift_endpoint(body: Features13):
    if len(body.features) != 13:
        raise HTTPException(400, f"Expected 13 features, got {len(body.features)}")
    return predict_deadlift(body.features)

@app.post("/predict/benchpress", response_model=FormPrediction)
async def predict_benchpress_endpoint(body: Features13):
    if len(body.features) != 13:
        raise HTTPException(400, f"Expected 13 features, got {len(body.features)}")
    return predict_benchpress(body.features)

@app.post("/predict/squat", response_model=FormPrediction)
async def predict_squat_endpoint(body: SquatFeatures):
    return predict_squat(body.model_dump())
