"""
MODULE 2: Feature Extractor
-----------------------------
Extracts 24 features per video (18 visual + 6 audio):

VISUAL (18):
  1.  avg_brightness         - average brightness
  2.  brightness_std         - brightness variation (AI = too stable)
  3.  avg_saturation         - color richness
  4.  saturation_std         - saturation variation
  5.  avg_edge_density       - edge sharpness
  6.  edge_std               - edge consistency
  7.  avg_blur_score         - frame sharpness
  8.  blur_std               - blur variation
  9.  motion_score           - movement between frames
  10. motion_std             - motion uniformity (AI = too smooth)
  11. scene_cut_count        - number of scene changes
  12. temporal_consistency   - frame similarity (AI = too similar)
  13. color_diversity        - color variety
  14. face_region_smoothness - skin smoothness (AI = too smooth)
  15. fps                    - frames per second
  16. duration               - video length
  17. resolution_score       - resolution
  18. aspect_ratio           - width/height

AUDIO (6):
  19. audio_energy           - overall loudness level
  20. audio_energy_std       - energy variation (AI = too uniform)
  21. silence_ratio          - proportion of silence (AI = less natural silence)
  22. spectral_centroid_mean - brightness of audio (AI audio = different freq profile)
  23. lipsync_score          - mouth motion vs audio energy correlation
  24. zero_crossing_rate     - audio texture (AI audio = unnaturally smooth)
"""

import cv2
import numpy as np
import os
import csv
import sys
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR  = os.path.join(BASE_DIR, "downloads")
DATA_DIR      = os.path.join(BASE_DIR, "data")
DATASET_CSV   = os.path.join(DATA_DIR, "dataset.csv")
FEATURES_CSV  = os.path.join(DATA_DIR, "features.csv")

FEATURE_COLS = [
    "filename", "is_ai",
    # Visual
    "avg_brightness", "brightness_std",
    "avg_saturation", "saturation_std",
    "avg_edge_density", "edge_std",
    "avg_blur_score", "blur_std",
    "motion_score", "motion_std",
    "scene_cut_count", "temporal_consistency",
    "color_diversity", "face_region_smoothness",
    "fps", "duration", "resolution_score", "aspect_ratio",
    # Audio
    "audio_energy", "audio_energy_std",
    "silence_ratio", "spectral_centroid_mean",
    "lipsync_score", "zero_crossing_rate",
]


# ── Visual feature functions ───────────────────────────────────────────────────

def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def get_saturation(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))

def get_edge_density(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / edges.size)

def get_blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def get_motion(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))

def get_temporal_similarity(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(1.0 - (np.mean(diff) / 255.0))

def get_color_diversity(frame):
    small = cv2.resize(frame, (64, 64))
    pixels = small.reshape(-1, 3)
    unique = len(np.unique(pixels, axis=0))
    return float(unique / (64 * 64))

def get_face_smoothness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    skin_pixels = gray[mask > 0]
    if len(skin_pixels) < 100:
        return 0.0
    return float(np.std(skin_pixels))

def get_mouth_motion(frame):
    """Estimate motion in the lower-center region (mouth area) of the frame."""
    h, w = frame.shape[:2]
    # Lower 1/3, center 1/3 = approximate mouth region
    mouth_region = frame[int(h*0.55):int(h*0.85), int(w*0.3):int(w*0.7)]
    if mouth_region.size == 0:
        return 0.0
    gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Audio feature functions ────────────────────────────────────────────────────

def extract_audio_features(video_path: str) -> dict:
    """
    Extract audio from video and compute audio features.
    Uses librosa if available, falls back to zeros if not.
    """
    defaults = {
        "audio_energy":          0.0,
        "audio_energy_std":      0.0,
        "silence_ratio":         0.0,
        "spectral_centroid_mean":0.0,
        "zero_crossing_rate":    0.0,
    }

    try:
        import librosa
        import soundfile as sf
        import tempfile
        import subprocess

        # Extract audio from video to a temp wav file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "22050", "-ac", "1", tmp_path],
            capture_output=True, timeout=30
        )

        if result.returncode != 0 or not os.path.exists(tmp_path):
            return defaults

        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        os.unlink(tmp_path)

        if len(y) == 0:
            return defaults

        # RMS energy per frame
        rms        = librosa.feature.rms(y=y)[0]
        energy     = float(np.mean(rms))
        energy_std = float(np.std(rms))

        # Silence ratio — frames where energy < 5% of max
        silence_threshold = np.max(rms) * 0.05
        silence_ratio     = float(np.sum(rms < silence_threshold) / len(rms))

        # Spectral centroid — "brightness" of audio
        centroid      = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = float(np.mean(centroid))

        # Zero crossing rate — audio texture
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))

        return {
            "audio_energy":          energy,
            "audio_energy_std":      energy_std,
            "silence_ratio":         silence_ratio,
            "spectral_centroid_mean":centroid_mean,
            "zero_crossing_rate":    zcr_mean,
        }

    except Exception:
        return defaults


def compute_lipsync_score(mouth_motion_vals: list, audio_energy_vals: list) -> float:
    """
    Lip-sync score: correlation between mouth movement and audio energy.
    High correlation = good lip sync = more likely real.
    Low/no correlation = bad lip sync = more likely AI.
    Returns value between -1 and 1.
    """
    if len(mouth_motion_vals) < 5 or len(audio_energy_vals) < 5:
        return 0.0

    # Match lengths
    min_len = min(len(mouth_motion_vals), len(audio_energy_vals))
    m = np.array(mouth_motion_vals[:min_len])
    a = np.array(audio_energy_vals[:min_len])

    # Avoid division by zero
    if np.std(m) == 0 or np.std(a) == 0:
        return 0.0

    correlation = np.corrcoef(m, a)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_features_from_video(video_path: str, sample_rate: int = 20) -> dict | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps if fps > 0 else 0

    brightness_vals, saturation_vals = [], []
    edge_vals, blur_vals             = [], []
    motion_vals, temporal_vals       = [], []
    color_div_vals, face_smooth_vals = [], []
    mouth_motion_vals                = []
    scene_cuts = 0

    prev_frame = None
    prev_gray  = None
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            frame_small = cv2.resize(frame, (320, 180))

            brightness_vals.append(get_brightness(frame_small))
            saturation_vals.append(get_saturation(frame_small))
            edge_vals.append(get_edge_density(frame_small))
            blur_vals.append(get_blur_score(frame_small))
            color_div_vals.append(get_color_diversity(frame_small))
            face_smooth_vals.append(get_face_smoothness(frame_small))
            mouth_motion_vals.append(get_mouth_motion(frame_small))

            curr_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                motion_vals.append(get_motion(prev_gray, curr_gray))
                sim = get_temporal_similarity(prev_frame, frame_small)
                temporal_vals.append(sim)
                brightness_diff = abs(brightness_vals[-1] - brightness_vals[-2])
                if brightness_diff > 40:
                    scene_cuts += 1

            prev_frame = frame_small.copy()
            prev_gray  = curr_gray.copy()

        frame_idx += 1

    cap.release()

    if not brightness_vals:
        return None

    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst):  return float(np.std(lst))  if lst else 0.0

    # Audio features
    audio_feats = extract_audio_features(video_path)

    # Lip-sync: correlate mouth motion with audio energy per-frame
    # We sample audio energy at same rate as frames
    audio_energy_sampled = []
    try:
        import librosa
        import tempfile, subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "22050", "-ac", "1", tmp_path],
            capture_output=True, timeout=30
        )
        if res.returncode == 0:
            y, sr = librosa.load(tmp_path, sr=22050, mono=True)
            os.unlink(tmp_path)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            # Resample rms to match number of sampled frames
            if len(rms) > 0:
                indices = np.linspace(0, len(rms)-1, len(mouth_motion_vals)).astype(int)
                audio_energy_sampled = rms[indices].tolist()
    except Exception:
        pass

    lipsync = compute_lipsync_score(mouth_motion_vals, audio_energy_sampled)

    return {
        # Visual
        "avg_brightness":        safe_mean(brightness_vals),
        "brightness_std":        safe_std(brightness_vals),
        "avg_saturation":        safe_mean(saturation_vals),
        "saturation_std":        safe_std(saturation_vals),
        "avg_edge_density":      safe_mean(edge_vals),
        "edge_std":              safe_std(edge_vals),
        "avg_blur_score":        safe_mean(blur_vals),
        "blur_std":              safe_std(blur_vals),
        "motion_score":          safe_mean(motion_vals),
        "motion_std":            safe_std(motion_vals),
        "scene_cut_count":       float(scene_cuts),
        "temporal_consistency":  safe_mean(temporal_vals),
        "color_diversity":       safe_mean(color_div_vals),
        "face_region_smoothness":safe_mean(face_smooth_vals),
        "fps":                   float(fps),
        "duration":              float(duration),
        "resolution_score":      float((width * height) / 1_000_000),
        "aspect_ratio":          float(width / height) if height > 0 else 0.0,
        # Audio
        "audio_energy":          audio_feats["audio_energy"],
        "audio_energy_std":      audio_feats["audio_energy_std"],
        "silence_ratio":         audio_feats["silence_ratio"],
        "spectral_centroid_mean":audio_feats["spectral_centroid_mean"],
        "lipsync_score":         lipsync,
        "zero_crossing_rate":    audio_feats["zero_crossing_rate"],
    }


# ── Run ────────────────────────────────────────────────────────────────────────

def run():
    print("\n========================================")
    print("   AI Reel Detector — Extractor Module")
    print("========================================\n")

    if not os.path.exists(DATASET_CSV):
        print("[✗] dataset.csv not found. Run the downloader first.")
        return

    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labeled = [r for r in rows if r.get("is_ai", "").strip() in ["0", "1"]]
    if not labeled:
        print("[✗] No labeled rows found in dataset.csv.")
        return

    print(f"Found {len(labeled)} labeled videos to process.\n")

    already_done = set()
    if os.path.exists(FEATURES_CSV):
        with open(FEATURES_CSV, "r", encoding="utf-8") as f:
            already_done = {r["filename"] for r in csv.DictReader(f)}

    file_exists = os.path.exists(FEATURES_CSV)
    out_file    = open(FEATURES_CSV, "a", newline="", encoding="utf-8")
    writer      = csv.DictWriter(out_file, fieldnames=FEATURE_COLS)
    if not file_exists:
        writer.writeheader()

    success, skipped, failed = 0, 0, 0

    for row in tqdm(labeled, desc="Extracting features", unit="video"):
        filename = row["filename"].strip()
        is_ai    = row["is_ai"].strip()

        if filename in already_done:
            skipped += 1
            continue

        video_path = os.path.join(DOWNLOAD_DIR, filename)
        if not os.path.exists(video_path):
            print(f"\n  [✗] File not found: {filename}")
            failed += 1
            continue

        features = extract_features_from_video(video_path)
        if features is None:
            print(f"\n  [✗] Could not read video: {filename}")
            failed += 1
            continue

        writer.writerow({"filename": filename, "is_ai": is_ai, **features})
        out_file.flush()
        success += 1

    out_file.close()

    print(f"\n========================================")
    print(f"  Done!")
    print(f"  Extracted : {success} videos")
    print(f"  Skipped   : {skipped} (already done)")
    print(f"  Failed    : {failed}")
    print(f"  Features  → {FEATURES_CSV}")
    print(f"\n  Next step: Run Module 3 to train the model!")
    print("========================================\n")


if __name__ == "__main__":
    run()
