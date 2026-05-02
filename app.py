"""
ReelOrReal — FastAPI Backend
Public: Detect + Download
Admin: History + Correct/Wrong + Auto retrain
"""

import os
import sys
import csv
import json
import warnings
import subprocess
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import yt_dlp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.extractor import extract_features_from_video

import psycopg2

def get_db():
    return psycopg2.connect(os.environ.get("DATABASE_URL"))

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            link TEXT,
            filename TEXT,
            prediction INTEGER,
            label TEXT,
            confidence REAL,
            features JSONB,
            feedback TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "detector.pkl")
TEMP_DIR      = os.path.join(BASE_DIR, "downloads", "temp")
COOKIES_PATH  = os.path.join(BASE_DIR, "cookies.txt")
# Write cookies from env variable if present
_cookies_env = os.environ.get("COOKIES_CONTENT")
if _cookies_env and not os.path.exists(COOKIES_PATH):
    with open(COOKIES_PATH, "w") as f:
        f.write(_cookies_env)
DATA_DIR      = os.path.join(BASE_DIR, "data")
HISTORY_FILE  = os.path.join(DATA_DIR, "history.json")
DATASET_CSV   = os.path.join(DATA_DIR, "dataset.csv")
FEATURES_CSV  = os.path.join(DATA_DIR, "features.csv")

ADMIN_CODE    = "ROR@X9#mK2$vP7!qZ"  # Secret admin code

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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

app       = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


# ── History helpers ────────────────────────────────────────────────────────────
def load_history() -> list:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT row_to_json(history) FROM history ORDER BY timestamp DESC")
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows

def save_history(history: list):
    pass  # no longer needed, kept for compatibility

def add_to_history(entry: dict):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history (id, link, filename, prediction, label, confidence, features, feedback, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET feedback = EXCLUDED.feedback
    """, (
        entry.get("id"),
        entry.get("link"),
        entry.get("filename"),
        entry.get("prediction"),
        entry.get("label"),
        entry.get("confidence"),
        json.dumps(entry.get("features", {})),
        entry.get("feedback"),
        entry.get("timestamp")
    ))
    conn.commit()
    cur.close()
    conn.close()


# ── Download helper ────────────────────────────────────────────────────────────
def download_reel(link: str, filename: str = None) -> str | None:
    if not filename:
        filename = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    out_template = os.path.join(TEMP_DIR, f"{filename}.%(ext)s")

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
            prepared = ydl.prepare_filename(info)
            base     = os.path.splitext(prepared)[0]
            for ext in [".mp4", ".mkv", ".webm", ".mov"]:
                if os.path.exists(base + ext):
                    return base + ext
            return prepared if os.path.exists(prepared) else None
    except Exception:
        return None


# ── Save correction to dataset ─────────────────────────────────────────────────
def save_correction_and_retrain(link: str, filename: str, correct_label: int, features: dict):
    """Save corrected entry to dataset + features, then retrain."""

    # dataset.csv
    exists = os.path.exists(DATASET_CSV)
    with open(DATASET_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "link", "is_ai", "downloaded_at"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "filename":      filename,
            "link":          link,
            "is_ai":         correct_label,
            "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # features.csv
    all_cols    = ["filename", "is_ai"] + FEATURE_COLS
    feat_exists = os.path.exists(FEATURES_CSV)
    with open(FEATURES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        if not feat_exists:
            writer.writeheader()
        row = {"filename": filename, "is_ai": correct_label}
        row.update({col: features.get(col, 0.0) for col in FEATURE_COLS})
        writer.writerow(row)

    # Retrain
    retrain_model()


def retrain_model():
    """Retrain ensemble model from features.csv."""
    global model
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv(FEATURES_CSV)
        df = df.dropna(subset=["is_ai"] + FEATURE_COLS)
        df = df[df["is_ai"].isin([0, 1])]

        if len(df) < 10:
            return

        X = df[FEATURE_COLS].values.astype(float)
        y = df["is_ai"].values.astype(int)

        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft"
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ensemble)
        ])
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_PATH)
        model = pipeline
    except Exception as e:
        print(f"Retrain failed: {e}")


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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename  = f"reel_{timestamp}.mp4"

    video_path = download_reel(link, f"reel_{timestamp}")
    if not video_path:
        return JSONResponse({"error": "Could not download this reel. It may have been deleted or is unavailable."}, status_code=400)

    features = extract_features_from_video(video_path)

    if not features:
        try:
            os.remove(video_path)
        except Exception:
            pass
        return JSONResponse({"error": "Could not analyze this video."}, status_code=500)

    X          = np.array([[features[col] for col in FEATURE_COLS]])
    prediction = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]
    confidence = round(float(proba[prediction]) * 100, 1)
    entry_id   = timestamp

    # Save to history
    add_to_history({
        "id":         entry_id,
        "link":       link,
        "filename":   filename,
        "prediction": prediction,
        "label":      "AI Generated" if prediction == 1 else "Real Video",
        "confidence": confidence,
        "features":   features,
        "feedback":   None,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Keep video for download
    actual_filename = os.path.basename(video_path)

    return JSONResponse({
        "prediction": prediction,
        "label":      "AI Generated" if prediction == 1 else "Real Video",
        "confidence": confidence,
        "entry_id":   entry_id,
        "filename":   actual_filename,
    })


@app.get("/download/{filename}")
async def download(filename: str):
    """Serve video file for download."""
    # Security: only allow files in temp dir
    safe_filename = os.path.basename(filename)
    file_path     = os.path.join(TEMP_DIR, safe_filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found."}, status_code=404)
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=safe_filename,
        headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
    )


@app.delete("/cleanup/{filename}")
async def cleanup(filename: str):
    """Delete temp file after user downloads."""
    safe_filename = os.path.basename(filename)
    file_path     = os.path.join(TEMP_DIR, safe_filename)
    try:
        os.remove(file_path)
    except Exception:
        pass
    return JSONResponse({"status": "ok"})


# ── Admin routes ───────────────────────────────────────────────────────────────
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, code: str = ""):
    if code != ADMIN_CODE:
        return templates.TemplateResponse("admin_login.html", {"request": request, "error": code != ""})
    history = load_history()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "history": history,
        "code":    ADMIN_CODE,
    })


@app.post("/admin/feedback")
async def admin_feedback(
    background_tasks: BackgroundTasks,
    entry_id: str    = Form(...),
    correct:  str    = Form(...),  # "yes" or "no"
    code:     str    = Form(...),
):
    if code != ADMIN_CODE:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    history = load_history()
    entry   = next((h for h in history if h["id"] == entry_id), None)

    if not entry:
        return JSONResponse({"error": "Entry not found"}, status_code=404)

    if correct == "yes":
        entry["feedback"] = "correct"
        save_history(history)
        return JSONResponse({"status": "marked_correct"})

    else:
        # Model was wrong — flip label and retrain
        correct_label         = 1 if entry["prediction"] == 0 else 0
        entry["feedback"]     = "corrected"
        entry["correct_label"]= correct_label
        save_history(history)

        # Retrain in background
        background_tasks.add_task(
            save_correction_and_retrain,
            entry["link"],
            entry["filename"],
            correct_label,
            entry["features"],
        )

        return JSONResponse({"status": "corrected_retraining"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
