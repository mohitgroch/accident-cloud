"""
Accident Detection Cloud API v3.0 — Async Job Queue
Member 4 — Cloud & AI Deployment

Endpoints:
  POST /predict          — Submit video → returns job_id instantly
  GET  /result/{job_id} — Poll for result
  POST /api/v1/detect    — Contract B (sync, for Member 1 fail-safe)
  GET  /health           — Health check
  GET  /                 — Dashboard
"""

import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

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
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "model/best.pt"))
THRESHOLD       = float(os.getenv("DECISION_THRESHOLD", "0.50"))
CONSEC_WINDOWS  = int(os.getenv("CONSECUTIVE_WINDOWS", "2"))
GDRIVE_FILE_ID  = os.getenv("MODEL_GDRIVE_FILE_ID", "")

# ---------------------------------------------------------------------------
# In-memory job store  {job_id -> dict}
# ---------------------------------------------------------------------------
JOBS: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Model globals
# ---------------------------------------------------------------------------
PREDICTOR        = None
DEVICE           = None
MODEL_LOAD_ERROR = None


# ---------------------------------------------------------------------------
# Model download + load
# ---------------------------------------------------------------------------

def _download_from_gdrive():
    if not GDRIVE_FILE_ID:
        return
    if CHECKPOINT_PATH.exists():
        logger.info(f"Checkpoint already at {CHECKPOINT_PATH}, skipping download.")
        return
    logger.info(f"Downloading model from Google Drive ({GDRIVE_FILE_ID}) ...")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import gdown
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        str(CHECKPOINT_PATH),
        quiet=False,
    )
    logger.info(f"Downloaded {CHECKPOINT_PATH.stat().st_size // 1024} KB")


def _load_model():
    global PREDICTOR, DEVICE, MODEL_LOAD_ERROR
    try:
        _download_from_gdrive()

        # Ensure root-level `inference/` package is importable
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from inference.model_loader import load_accident_model
        from inference.predictor   import AccidentPredictor

        logger.info(f"Loading model from {CHECKPOINT_PATH} ...")
        model, DEVICE  = load_accident_model(CHECKPOINT_PATH)
        PREDICTOR       = AccidentPredictor(model, DEVICE)
        logger.info(f"Model loaded on {DEVICE}")

    except FileNotFoundError:
        MODEL_LOAD_ERROR = f"Checkpoint not found: {CHECKPOINT_PATH}"
        logger.warning(MODEL_LOAD_ERROR + " — STUB mode")
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        logger.error(f"Model load error: {exc} — STUB mode")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _stub_result(video_path: str) -> dict:
    return {
        "video_path": video_path, "prediction": "NORMAL",
        "probabilities": [0.1], "windows": 1,
        "preprocessing_time": 0.0, "inference_time": 0.0,
        "total_processing_time": 0.0,
    }


def _run_inference(tmp_path: str) -> dict:
    if PREDICTOR is not None:
        return PREDICTOR.predict(tmp_path, threshold=THRESHOLD,
                                 consecutive_windows=CONSEC_WINDOWS)
    return _stub_result(tmp_path)


def _inference_worker(job_id: str, tmp_path: str):
    """Runs in a background thread. Updates JOBS[job_id] when done."""
    logger.info(f"[{job_id}] Background inference started")
    t0 = time.perf_counter()
    try:
        result = _run_inference(tmp_path)
        elapsed = round(time.perf_counter() - t0, 2)

        accident   = result["prediction"] == "ACCIDENT"
        probs      = result["probabilities"]
        risk_score = round(float(max(probs)) if probs else 0.0, 4)

        JOBS[job_id] = {
            "status":               "done",
            "success":              True,
            "accident_detected":    accident,
            "risk_score":           risk_score,
            "message":              "Potential accident detected" if accident else "Normal driving",
            "inference_time_seconds": elapsed,
            "windows_analysed":     result["windows"],
            "probabilities":        probs,
        }
        logger.info(f"[{job_id}] Done in {elapsed}s | {result['prediction']} | risk={risk_score}")

    except Exception as exc:
        logger.error(f"[{job_id}] Inference error: {exc}")
        JOBS[job_id] = {
            "status":  "error",
            "success": False,
            "message": str(exc),
        }
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Accident Detection Cloud API",
    version="3.0.0",
    description="Async MobileNetV2+GRU inference for vehicle accident detection",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# POST /predict  — async job submission (returns instantly)
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """
    Submit a video for inference. Returns job_id immediately.
    Poll GET /result/{job_id} for the result.

    Response:
      { "job_id": "abc123", "status": "queued",
        "poll_url": "/result/abc123", "message": "Processing started" }
    """
    job_id = str(uuid.uuid4())[:8]
    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"

    logger.info(f"[{job_id}] /predict received: {video.filename}")

    # Save upload to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(500, f"Failed to save upload: {exc}")

    # Register job and start background thread
    JOBS[job_id] = {"status": "processing"}
    t = threading.Thread(target=_inference_worker, args=(job_id, tmp_path), daemon=True)
    t.start()

    return {
        "job_id":   job_id,
        "status":   "queued",
        "poll_url": f"/result/{job_id}",
        "message":  "Processing started. Poll /result/{job_id} for results.",
    }


# ---------------------------------------------------------------------------
# GET /result/{job_id}  — poll for inference result
# ---------------------------------------------------------------------------
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Poll for inference result.
    Returns "processing" until done, then full result.

    Example response when done:
      { "status": "done", "accident_detected": true,
        "risk_score": 0.87, "inference_time_seconds": 45.2 }
    """
    if job_id not in JOBS:
        raise HTTPException(404, f"Job {job_id} not found")
    return JOBS[job_id]


# ---------------------------------------------------------------------------
# POST /api/v1/detect  — Contract B (sync, for Member 1 fail-safe)
# This endpoint starts inference in background but returns queued status.
# Member 1 should poll /result/{job_id} for the final result.
# ---------------------------------------------------------------------------
@app.post("/api/v1/detect")
async def detect_contract_b(
    video:      UploadFile = File(...),
    event_id:   str   = Form(default=None),
    peak_accel: float = Form(default=0.0),
    latitude:   float = Form(default=0.0),
    longitude:  float = Form(default=0.0),
):
    """
    Contract B endpoint for Member 1 fail-safe logic.
    Starts async inference, returns job_id to poll.
    """
    eid    = event_id or str(uuid.uuid4())[:8]
    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"

    logger.info(f"[{eid}] /api/v1/detect received | peak_accel={peak_accel}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
    except Exception as exc:
        return JSONResponse(500, content={
            "event_id": eid, "accident_confirmed": False,
            "confidence": 0.0, "status": "error",
        })

    JOBS[eid] = {"status": "processing"}
    t = threading.Thread(target=_inference_worker, args=(eid, tmp_path), daemon=True)
    t.start()

    return {
        "event_id":  eid,
        "status":    "queued",
        "poll_url":  f"/result/{eid}",
        "message":   "Processing started. Poll /result/{eid} for accident_confirmed.",
        "peak_accel": peak_accel,
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status":             "ok",
        "model_loaded":       PREDICTOR is not None,
        "model_error":        MODEL_LOAD_ERROR,
        "device":             str(DEVICE) if DEVICE else "none",
        "checkpoint":         str(CHECKPOINT_PATH),
        "threshold":          THRESHOLD,
        "consecutive_windows": CONSEC_WINDOWS,
        "active_jobs":        len([j for j in JOBS.values() if j.get("status") == "processing"]),
    }


# ---------------------------------------------------------------------------
# GET /  — Dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    model_status = (
        f"Loaded on <b>{DEVICE}</b>" if PREDICTOR
        else f"Stub mode — {MODEL_LOAD_ERROR}"
    )
    return f"""<!DOCTYPE html>
<html><head><title>Accident Detection API v3</title>
<style>
body{{font-family:sans-serif;max-width:750px;margin:40px auto;padding:0 20px;line-height:1.6}}
code,pre{{background:#f4f4f4;padding:2px 6px;border-radius:4px;font-size:0.9em}}
pre{{display:block;padding:12px;white-space:pre-wrap}}
h3{{margin-bottom:4px}}
</style></head><body>
<h1>Accident Detection Cloud API v3.0</h1>
<p>Model: {model_status}</p>
<hr>
<h2>How to Use (Async Flow)</h2>
<pre>
Step 1 — Submit video:
  POST /predict
  Form field: video (mp4 file)
  Returns:  {{"job_id": "abc123", "poll_url": "/result/abc123"}}

Step 2 — Poll for result:
  GET /result/abc123
  Returns "processing" until done, then:
  {{
    "status": "done",
    "accident_detected": true,
    "risk_score": 0.87,
    "message": "Potential accident detected",
    "inference_time_seconds": 45.2,
    "windows_analysed": 3
  }}
</pre>
<h3>Contract B (Member 1 fail-safe)</h3>
<pre>POST /api/v1/detect  — same async flow, event_id as job_id</pre>
<h3>Health</h3>
<pre>GET /health  — model status, active jobs</pre>
<p><a href="/docs">Interactive API Docs (Swagger UI)</a></p>
</body></html>"""
