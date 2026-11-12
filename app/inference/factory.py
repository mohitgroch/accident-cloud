import asyncio
from typing import Optional, Dict, Any, Tuple
from app.inference.base import BaseInferenceEngine
from app.inference.stub_engine import StubInferenceEngine
from app.inference.model_engine import ModelInferenceEngine
from app.inference.colab_model_engine import ColabModelInferenceEngine
from app.core.config import settings
from app.core.logger import get_logger

class InferenceManager:
    """
    Manages active inference engine and enforces the Pi timeout SLA.
    Supports:
      - 'stub' : Phase 1 testing engine
      - 'colab': Member 3's MobileNetTemporal / Colab tunnel integration
      - 'model': Generic PyTorch/YOLO engine
    """

    def __init__(self):
        self._mode = settings.INFERENCE_MODE.lower()
        self._stub_engine = StubInferenceEngine()
        self._colab_engine = None
        self._model_engine = None
        self._forced_override: Optional[Dict[str, Any]] = None

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, new_mode: str):
        valid_modes = ("stub", "colab", "model")
        if new_mode.lower() not in valid_modes:
            raise ValueError(f"Invalid mode '{new_mode}'. Must be one of {valid_modes}")
        self._mode = new_mode.lower()
        if self._mode == "colab" and self._colab_engine is None:
            self._colab_engine = ColabModelInferenceEngine()
        elif self._mode == "model" and self._model_engine is None:
            self._model_engine = ModelInferenceEngine()
        get_logger("SYSTEM").info(f"[InferenceManager] Switched inference mode to: {self._mode}")

    def set_forced_override(self, override: Optional[Dict[str, Any]]):
        """Allows test suites to force a specific verdict or status."""
        self._forced_override = override

    def get_active_engine(self) -> BaseInferenceEngine:
        if self._mode == "colab":
            if self._colab_engine is None:
                self._colab_engine = ColabModelInferenceEngine()
            return self._colab_engine
        elif self._mode == "model":
            if self._model_engine is None:
                self._model_engine = ModelInferenceEngine()
            return self._model_engine
        return self._stub_engine

    async def execute_inference_with_timeout(
        self,
        video_path: str,
        sensor_summary: Optional[Dict[str, Any]] = None,
        gps_info: Optional[Dict[str, Any]] = None,
        event_id: str = "unknown",
        timeout_seconds: Optional[float] = None
    ) -> Tuple[bool, float, str, Optional[str]]:
        """
        Executes prediction with timeout enforcement.
        
        Returns:
            accident_confirmed (bool)
            confidence (float)
            status (str): "ok", "error", or "timeout"
            diagnostic (str): Contextual message
        """
        logger = get_logger(event_id)
        effective_timeout = timeout_seconds or settings.TIMEOUT_SECONDS

        # 1. Check test overrides
        if self._forced_override:
            status = self._forced_override.get("status", "ok")
            verdict = self._forced_override.get("accident_confirmed", False)
            conf = self._forced_override.get("confidence", 0.0)
            return verdict, conf, status, "Forced test override active"

        engine = self.get_active_engine()
        logger.info(f"[InferenceManager] Dispatching to {engine.__class__.__name__} (timeout={effective_timeout}s)")

        try:
            # Shielded timeout execution
            task = engine.predict(
                video_path=video_path,
                sensor_summary=sensor_summary,
                gps_info=gps_info,
                event_id=event_id
            )
            verdict, confidence, diag = await asyncio.wait_for(task, timeout=effective_timeout)
            return verdict, confidence, "ok", diag

        except asyncio.TimeoutError:
            logger.error(
                f"[InferenceManager] TIMEOUT EXCEEDED: Processing for {event_id} took > {effective_timeout}s. "
                "Returning timeout status for Member 1 fail-safe."
            )
            return False, 0.0, "timeout", f"Inference exceeded {effective_timeout}s deadline"

        except Exception as e:
            logger.exception(f"[InferenceManager] INFERENCE FAILURE on {event_id}: {e}")
            return False, 0.0, "error", f"Inference engine failure: {str(e)}"

    def get_info(self) -> Dict[str, Any]:
        engine = self.get_active_engine()
        info = engine.get_info()
        info["active_mode"] = self._mode
        info["timeout_seconds"] = settings.TIMEOUT_SECONDS
        info["forced_override"] = self._forced_override is not None
        return info

# Global inference manager singleton
inference_manager = InferenceManager()

