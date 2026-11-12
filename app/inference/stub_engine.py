import asyncio
import hashlib
import os
import random
from typing import Optional, Dict, Any, Tuple
from app.inference.base import BaseInferenceEngine
from app.core.config import settings
from app.core.logger import get_logger

class StubInferenceEngine(BaseInferenceEngine):
    """
    Phase 1: Stub / Dummy Inference Engine.
    
    Enables Member 2 to upload real camera clips and Member 1 to verify
    their alert and fail-safe logic before Member 3's real model is ready.
    
    Deterministic Simulation Rules (for easy testing):
      - If event_id contains 'crash', 'accident', 'collision' -> returns (True, ~0.94)
      - If event_id contains 'pothole', 'brake', 'normal', 'reject' -> returns (False, ~0.18)
      - If event_id contains 'timeout' -> sleeps longer than timeout to test fail-safe
      - If event_id contains 'fail' or 'crash_err' -> raises exception to test error handling
      - Otherwise: uses pseudo-random hash or MOCK_DEFAULT_VERDICT setting.
    """

    def __init__(self):
        self.latency = settings.MOCK_INFERENCE_LATENCY
        self.default_verdict = settings.MOCK_DEFAULT_VERDICT

    async def predict(
        self,
        video_path: str,
        sensor_summary: Optional[Dict[str, Any]] = None,
        gps_info: Optional[Dict[str, Any]] = None,
        event_id: str = "unknown"
    ) -> Tuple[bool, float, Optional[str]]:
        logger = get_logger(event_id)
        logger.info(f"[StubEngine] Simulating inference for event {event_id} (video: {video_path})")

        # 1. Check for intentional test triggers
        lower_id = event_id.lower()

        # Intentional timeout simulation
        if "timeout" in lower_id:
            delay = settings.TIMEOUT_SECONDS + 2.0
            logger.warning(f"[StubEngine] Test trigger 'timeout' detected. Sleeping for {delay}s...")
            await asyncio.sleep(delay)

        # Intentional error simulation
        if "fail_test" in lower_id or "error_test" in lower_id:
            logger.error("[StubEngine] Test trigger 'fail_test' detected. Raising simulated exception.")
            raise RuntimeError("Simulated inference engine crash for fail-safe testing.")

        # Simulate normal model compute latency
        if self.latency > 0:
            await asyncio.sleep(self.latency)

        # 2. Check for explicit keyword overrides
        if any(k in lower_id for k in ["crash", "accident", "impact", "collision", "hit"]):
            confidence = round(random.uniform(0.88, 0.98), 4)
            logger.info(f"[StubEngine] Keyword override: Confirmed accident (conf={confidence})")
            return True, confidence, "Stub: High-confidence collision pattern detected"

        if any(k in lower_id for k in ["pothole", "brake", "braking", "normal", "reject", "false_alarm"]):
            confidence = round(random.uniform(0.08, 0.32), 4)
            logger.info(f"[StubEngine] Keyword override: Rejected false alarm (conf={confidence})")
            return False, confidence, "Stub: Transient motion classified as pothole/braking (non-accident)"

        # 3. Default behavior based on file hash and sensor summary
        # If sensor summary shows extreme jerk/accel, tilt towards positive
        high_g_detected = False
        if sensor_summary and isinstance(sensor_summary, dict):
            peak_accel = sensor_summary.get("peak_accel")
            if peak_accel and float(peak_accel) > 25.0:  # e.g., > 2.5G
                high_g_detected = True

        if self.default_verdict == "true":
            verdict = True
            confidence = round(random.uniform(0.85, 0.96), 4)
        elif self.default_verdict == "false":
            verdict = False
            confidence = round(random.uniform(0.10, 0.35), 4)
        elif high_g_detected:
            verdict = True
            confidence = round(random.uniform(0.80, 0.95), 4)
        else:
            # Hash-based pseudo random for consistency across identical uploads
            file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 1000
            seed_val = int(hashlib.md5(f"{event_id}_{file_size}".encode()).hexdigest()[:6], 16)
            verdict = (seed_val % 100) < 50  # 50% split
            if verdict:
                confidence = round(0.70 + (seed_val % 28) / 100.0, 4)
            else:
                confidence = round(0.15 + (seed_val % 30) / 100.0, 4)

        logger.info(f"[StubEngine] Result for {event_id}: accident_confirmed={verdict}, confidence={confidence}")
        return verdict, confidence, f"Stub: Simulated verdict (high_g={high_g_detected})"

    def get_info(self) -> Dict[str, Any]:
        return {
            "engine": "StubInferenceEngine",
            "phase": "Phase 1 (Dummy/Stub)",
            "simulated_latency": self.latency,
            "default_verdict": self.default_verdict,
            "status": "ready"
        }

