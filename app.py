"""
RealOrReel — FastAPI Backend
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import yt_dlp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.extractor import extract_features_from_video

# ── Setup ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "detector.pkl")
TEMP_DIR     = os.path.join(BASE_DIR, "downloads", "predictions")
COOKIES_PATH = os.path.join(BASE_DIR, "cookies.txt")

os.makedirs(TEMP_DIR, exist_ok=True)

app       = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model once at startup
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "avg_brightness", "brightness_std",
    "avg_saturation", "saturation_std",
    "avg_edge_density", "edge_std",
    "avg_blur_score", "blur_std",
    "motion_score", "motion_std",
    "scene_cut_count", "temporal_consistency",
    "color_diversity", "face_region_smoothness",
    "fps", "duration", "resolution_score", "aspect_ratio",
    "audio_energy", "audio_energy_std",
    "silence_ratio", "spectral_centroid_mean",
    "lipsync_score", "zero_crossing_rate",
]


# ── Download ───────────────────────────────────────────────────────────────────
def download_reel(link: str) -> str | None:
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_template = os.path.join(TEMP_DIR, f"pred_{timestamp}.%(ext)s")

    ydl_opts = {
        "outtmpl":             out_template,
        "format":              "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet":               True,
        "no_warnings":         True,
        "cookiefile":          COOKIES_PATH if os.path.exists(COOKIES_PATH) else None,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info     = ydl.extract_info(link, download=True)
            filename = ydl.prepare_filename(info)
            base     = os.path.splitext(filename)[0]
            for ext in [".mp4", ".mkv", ".webm", ".mov"]:
                if os.path.exists(base + ext):
                    return base + ext
            return filename if os.path.exists(filename) else None
    except Exception:
        return None


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(link: str = Form(...)):
    if not model:
        return JSONResponse({"error": "Model not loaded."}, status_code=500)

    if "instagram.com" not in link:
        return JSONResponse({"error": "Please enter a valid Instagram Reel link."}, status_code=400)

    # Download
    video_path = download_reel(link)
    if not video_path:
        return JSONResponse({
            "error": "Could not download this reel. It may have been deleted or is unavailable."
        }, status_code=400)

    # Extract features
    features = extract_features_from_video(video_path)

    # Delete temp video
    try:
        os.remove(video_path)
    except Exception:
        pass

    if not features:
        return JSONResponse({"error": "Could not analyze this video."}, status_code=500)

    # Predict
    X          = np.array([[features[col] for col in FEATURE_COLS]])
    prediction = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]

    return JSONResponse({
        "prediction": prediction,
        "label":      "AI Generated" if prediction == 1 else "Real Video",
        "confidence": round(float(proba[prediction]) * 100, 1),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
