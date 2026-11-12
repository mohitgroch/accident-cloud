"""
Accident Detection Cloud API
Member 4 — Cloud & AI Deployment
BMSCE Major Project: AI-Assisted Smart Vehicle Accident Detection

Endpoints:
  POST /predict          — Primary endpoint (Member 3 spec + Pi integration)
  POST /api/v1/detect    — Contract B endpoint (Member 1 fail-safe logic)
  GET  /health           — Health check
  GET  /                 — Status dashboard
"""

import logging
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("accident_api")

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------
MODEL = None
DEVICE = None
PREDICTOR = None
MODEL_LOAD_ERROR = None

CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "model/best.pt"))
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.50"))
CONSECUTIVE_WINDOWS = int(os.getenv("CONSECUTIVE_WINDOWS", "2"))

# Google Drive file ID for best.pt (set via Render env var MODEL_GDRIVE_FILE_ID)
MODEL_GDRIVE_FILE_ID = os.getenv("MODEL_GDRIVE_FILE_ID", "")


def _download_model_from_gdrive():
    """Download best.pt from Google Drive if not already present."""
    if not MODEL_GDRIVE_FILE_ID:
        return  # No Drive ID configured — skip
    if CHECKPOINT_PATH.exists():
        logger.info(f"Checkpoint already present at {CHECKPOINT_PATH}, skipping download.")
        return

    logger.info(f"Downloading model from Google Drive (ID: {MODEL_GDRIVE_FILE_ID}) ...")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_FILE_ID}"
        gdown.download(url, str(CHECKPOINT_PATH), quiet=False)
        logger.info(f"Model downloaded to {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size // 1024} KB)")
    except Exception as exc:
        logger.error(f"Failed to download model from Google Drive: {exc}")
        raise


def _load_model():
    global MODEL, DEVICE, PREDICTOR, MODEL_LOAD_ERROR
    try:
        # Try to download model from Google Drive if not present
        _download_model_from_gdrive()

        # Add project root to path so `inference` package is importable
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from inference.model_loader import load_accident_model
        from inference.predictor import AccidentPredictor

        logger.info(f"Loading model from {CHECKPOINT_PATH} ...")
        MODEL, DEVICE = load_accident_model(CHECKPOINT_PATH)
        PREDICTOR = AccidentPredictor(MODEL, DEVICE)
        logger.info(f"Model loaded successfully on {DEVICE}")
    except FileNotFoundError:
        MODEL_LOAD_ERROR = f"Checkpoint not found: {CHECKPOINT_PATH}"
        logger.warning(MODEL_LOAD_ERROR + " — running in STUB mode")
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        logger.error(f"Model load failed: {exc} — running in STUB mode")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Accident Detection Cloud API",
    version="2.0.0",
    description="MobileNetV2+GRU inference service for vehicle accident detection",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_predict(video_path: str) -> dict:
    """Fallback when model weights are unavailable (returns NORMAL)."""
    return {
        "video_path": video_path,
        "prediction": "NORMAL",
        "probabilities": [0.1],
        "windows": 1,
        "preprocessing_time": 0.0,
        "inference_time": 0.0,
        "total_processing_time": 0.0,
    }


def _run_inference(tmp_path: str) -> dict:
    if PREDICTOR is not None:
        return PREDICTOR.predict(
            tmp_path,
            threshold=THRESHOLD,
            consecutive_windows=CONSECUTIVE_WINDOWS,
        )
    return _stub_predict(tmp_path)


# ---------------------------------------------------------------------------
# POST /predict  — Primary endpoint (Member 3 spec)
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """
    Accept a video file from the Raspberry Pi and return accident verdict.

    Request: multipart/form-data, field name = 'video'
    Response:
      {
        "success": true,
        "accident_detected": true,
        "risk_score": 0.91,
        "message": "Potential accident detected",
        "inference_time_seconds": 1.23,
        "windows_analysed": 5
      }
    """
    allowed = {"video/mp4", "video/avi", "video/x-msvideo", "video/quicktime",
               "video/x-matroska", "application/octet-stream"}
    if video.content_type and video.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {video.content_type}. Send a video file."
        )

    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    logger.info(f"[{request_id}] /predict received: {video.filename} ({video.content_type})")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
    except Exception as exc:
        logger.error(f"[{request_id}] Failed to save upload: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    try:
        result = _run_inference(tmp_path)
    except ValueError as exc:
        logger.warning(f"[{request_id}] Video processing error: {exc}")
        return JSONResponse(status_code=422, content={
            "success": False,
            "accident_detected": False,
            "risk_score": 0.0,
            "message": f"Video processing error: {exc}",
            "inference_time_seconds": round(time.perf_counter() - t0, 3),
            "windows_analysed": 0,
        })
    except Exception as exc:
        logger.error(f"[{request_id}] Inference error: {exc}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "accident_detected": False,
            "risk_score": 0.0,
            "message": f"Internal inference error: {exc}",
            "inference_time_seconds": round(time.perf_counter() - t0, 3),
            "windows_analysed": 0,
        })
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    accident_detected = result["prediction"] == "ACCIDENT"
    probs = result["probabilities"]
    risk_score = round(float(max(probs)) if probs else 0.0, 4)
    total_time = round(time.perf_counter() - t0, 3)

    logger.info(
        f"[{request_id}] Result: {result['prediction']} | "
        f"risk={risk_score} | windows={result['windows']} | "
        f"total={total_time}s"
    )

    return {
        "success": True,
        "accident_detected": accident_detected,
        "risk_score": risk_score,
        "message": "Potential accident detected" if accident_detected else "Normal driving",
        "inference_time_seconds": total_time,
        "windows_analysed": result["windows"],
    }


# ---------------------------------------------------------------------------
# POST /api/v1/detect  — Contract B endpoint (Member 1 fail-safe)
# ---------------------------------------------------------------------------
@app.post("/api/v1/detect")
async def detect_contract_b(
    video: UploadFile = File(...),
    event_id: str = Form(default=None),
    peak_accel: float = Form(default=0.0),
    latitude: float = Form(default=0.0),
    longitude: float = Form(default=0.0),
):
    """
    Contract B response for Member 1's fail-safe logic.
    Fields: event_id, accident_confirmed (bool), confidence (float 0-1), status (str)
    """
    eid = event_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"

    logger.info(f"[{eid}] /api/v1/detect received | peak_accel={peak_accel}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
    except Exception as exc:
        logger.error(f"[{eid}] Save failed: {exc}")
        return JSONResponse(status_code=500, content={
            "event_id": eid,
            "accident_confirmed": False,
            "confidence": 0.0,
            "status": "error",
        })

    try:
        result = _run_inference(tmp_path)
    except Exception as exc:
        logger.error(f"[{eid}] Inference failed: {exc}")
        return JSONResponse(status_code=500, content={
            "event_id": eid,
            "accident_confirmed": False,
            "confidence": 0.0,
            "status": "error",
        })
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    if elapsed > 5.0:
        logger.warning(f"[{eid}] Timeout SLA exceeded: {elapsed:.2f}s")
        return JSONResponse(status_code=504, content={
            "event_id": eid,
            "accident_confirmed": False,
            "confidence": 0.0,
            "status": "timeout",
        })

    accident_confirmed = result["prediction"] == "ACCIDENT"
    probs = result["probabilities"]
    confidence = round(float(max(probs)) if probs else 0.0, 4)

    # Sensor fusion: MPU6050 high-G boost
    if peak_accel > 25.0 and confidence > 0.3:
        confidence = min(1.0, confidence + 0.1)
        logger.info(f"[{eid}] Sensor fusion boost: peak_accel={peak_accel}")

    logger.info(f"[{eid}] Contract B: confirmed={accident_confirmed} confidence={confidence} elapsed={elapsed:.2f}s")

    return {
        "event_id": eid,
        "accident_confirmed": accident_confirmed,
        "confidence": confidence,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": PREDICTOR is not None,
        "model_error": MODEL_LOAD_ERROR,
        "device": str(DEVICE) if DEVICE else "none",
        "checkpoint": str(CHECKPOINT_PATH),
        "threshold": THRESHOLD,
        "consecutive_windows": CONSECUTIVE_WINDOWS,
    }


# ---------------------------------------------------------------------------
# GET /  — Status page
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    model_status = (
        f"Loaded on <b>{DEVICE}</b>"
        if PREDICTOR is not None
        else f"Stub mode (no weights) — {MODEL_LOAD_ERROR}"
    )
    return f"""<!DOCTYPE html>
<html><head><title>Accident Detection API</title>
<style>
body{{font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px}}
code{{background:#f4f4f4;padding:2px 6px;border-radius:4px}}
pre{{background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;overflow-x:auto}}
</style></head><body>
<h1>Accident Detection Cloud API v2.0</h1>
<p>Model: {model_status}</p>
<hr>
<h2>Endpoints</h2>
<h3>POST /predict</h3>
<p>Primary endpoint. Send video as <code>multipart/form-data</code>, field <code>video</code>.</p>
<pre>curl -X POST https://YOUR_URL/predict -F "video=@clip.mp4"</pre>
<pre>Response: {{"success":true,"accident_detected":true,"risk_score":0.91,"message":"Potential accident detected","inference_time_seconds":1.23,"windows_analysed":5}}</pre>
<h3>POST /api/v1/detect</h3>
<p>Contract B endpoint for Member 1 fail-safe logic.</p>
<pre>curl -X POST https://YOUR_URL/api/v1/detect -F "video=@clip.mp4" -F "event_id=evt_001" -F "peak_accel=28.5"</pre>
<pre>Response: {{"event_id":"evt_001","accident_confirmed":true,"confidence":0.91,"status":"ok"}}</pre>
<h3>GET /health</h3>
<p>Returns model load status and config.</p>
<p><a href="/docs">Interactive API Docs (Swagger UI)</a></p>
</body></html>"""
