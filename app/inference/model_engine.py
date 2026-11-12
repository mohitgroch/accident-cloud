import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from app.inference.base import BaseInferenceEngine
from app.core.config import settings
from app.core.logger import get_logger

class ModelInferenceEngine(BaseInferenceEngine):
    """
    Phase 2: Real AI Model Inference Engine.
    
    Loads Member 3's trained model (YOLOv8/YOLOv11, PyTorch, ONNX, or Custom Script)
    behind the exact same API contract.
    """

    def __init__(self):
        self.weights_path = Path(settings.MODEL_WEIGHTS_PATH)
        self.model_type = settings.MODEL_TYPE.lower()
        self.device = settings.MODEL_DEVICE
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.model = None
        self._load_model()

    def _load_model(self):
        logger = get_logger("SYSTEM")
        if not self.weights_path.exists():
            logger.warning(
                f"[ModelEngine] Weights file not found at '{self.weights_path}'. "
                "Engine initialized in fallback standby mode until Member 3 drops the model file into 'models/'."
            )
            return

        logger.info(f"[ModelEngine] Loading {self.model_type} model from '{self.weights_path}' on {self.device}...")
        try:
            if self.model_type == "yolo":
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(str(self.weights_path))
                    logger.info("[ModelEngine] Ultralytics YOLO model loaded successfully.")
                except ImportError:
                    logger.error("[ModelEngine] 'ultralytics' library not installed. Run 'pip install ultralytics'.")
            elif self.model_type in ("pytorch", "torch"):
                try:
                    import torch
                    self.model = torch.jit.load(str(self.weights_path), map_location=self.device)
                    self.model.eval()
                    logger.info("[ModelEngine] TorchScript model loaded successfully.")
                except ImportError:
                    logger.error("[ModelEngine] 'torch' library not installed.")
            elif self.model_type == "onnx":
                try:
                    import onnxruntime as ort
                    self.model = ort.InferenceSession(str(self.weights_path))
                    logger.info("[ModelEngine] ONNX Runtime session loaded successfully.")
                except ImportError:
                    logger.error("[ModelEngine] 'onnxruntime' library not installed.")
            else:
                logger.info(f"[ModelEngine] Generic/Custom model type: {self.model_type}")
        except Exception as e:
            logger.error(f"[ModelEngine] Error loading model weights: {e}")
            self.model = None

    def _extract_key_frames(self, video_path: str, max_frames: int = 16) -> List[Any]:
        """
        Extracts evenly spaced frames from the video clip for computer vision analysis.
        """
        frames = []
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # Read frame by frame
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    if len(frames) >= max_frames:
                        break
            else:
                step = max(1, total_frames // max_frames)
                for idx in range(0, total_frames, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames.append(frame)
                    if len(frames) >= max_frames:
                        break
            cap.release()
        except ImportError:
            pass
        except Exception as e:
            get_logger("SYSTEM").error(f"[ModelEngine] Video decoding error on {video_path}: {e}")
        return frames

    async def predict(
        self,
        video_path: str,
        sensor_summary: Optional[Dict[str, Any]] = None,
        gps_info: Optional[Dict[str, Any]] = None,
        event_id: str = "unknown"
    ) -> Tuple[bool, float, Optional[str]]:
        logger = get_logger(event_id)
        logger.info(f"[ModelEngine] Running real model inference for event {event_id}")

        # If model is not loaded (e.g. weights pending from Member 3)
        if self.model is None:
            logger.warning("[ModelEngine] Model weights not loaded. Falling back to heuristic baseline.")
            return await self._heuristic_baseline(video_path, sensor_summary, event_id)

        # Offload synchronous CV/PyTorch inference to worker thread so async event loop is never blocked
        loop = asyncio.get_running_loop()
        verdict, confidence, diag = await loop.run_in_executor(
            None,
            self._run_inference_sync,
            video_path,
            sensor_summary,
            gps_info,
            event_id
        )
        return verdict, confidence, diag

    def _run_inference_sync(
        self,
        video_path: str,
        sensor_summary: Optional[Dict[str, Any]],
        gps_info: Optional[Dict[str, Any]],
        event_id: str
    ) -> Tuple[bool, float, Optional[str]]:
        logger = get_logger(event_id)
        frames = self._extract_key_frames(video_path, max_frames=16)
        logger.info(f"[ModelEngine] Extracted {len(frames)} frames from video for inference.")

        if not frames:
            logger.warning("[ModelEngine] No frames extracted from video clip.")
            return False, 0.0, "Could not extract video frames"

        max_accident_conf = 0.0
        collision_detected = False

        if self.model_type == "yolo" and self.model is not None:
            # Ultralytics inference
            results = self.model(frames, verbose=False)
            for res in results:
                # Check for crash/accident class or bounding box overlaps
                # Depending on Member 3's class names: 'accident', 'crash', 'severe_damage'
                for box in res.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names.get(cls_id, "").lower()
                    if "accident" in cls_name or "crash" in cls_name or cls_id == 1:
                        if conf > max_accident_conf:
                            max_accident_conf = conf

        # Apply confidence threshold
        if max_accident_conf >= self.confidence_threshold:
            collision_detected = True

        # Sensor fusion bonus: if MPU6050 recorded massive peak jerk/g-force
        if sensor_summary and isinstance(sensor_summary, dict):
            peak_jerk = sensor_summary.get("peak_jerk")
            if peak_jerk and float(peak_jerk) > 50.0:
                max_accident_conf = min(1.0, max_accident_conf + 0.15)
                if max_accident_conf >= self.confidence_threshold:
                    collision_detected = True

        return collision_detected, round(max_accident_conf, 4), f"Model: {self.model_type} (frames={len(frames)})"

    async def _heuristic_baseline(
        self,
        video_path: str,
        sensor_summary: Optional[Dict[str, Any]],
        event_id: str
    ) -> Tuple[bool, float, Optional[str]]:
        """Standby baseline when weights file is not yet deployed."""
        await asyncio.sleep(0.1)
        # Check sensor summary
        peak_g = 0.0
        if sensor_summary and isinstance(sensor_summary, dict):
            peak_g = float(sensor_summary.get("peak_accel", 0.0) or 0.0)

        confirmed = peak_g > 30.0
        conf = 0.85 if confirmed else 0.20
        return confirmed, conf, "ModelStandby: Pending Member 3 weights file in models/"

    def get_info(self) -> Dict[str, Any]:
        return {
            "engine": "ModelInferenceEngine",
            "phase": "Phase 2 (Real Model)",
            "model_type": self.model_type,
            "weights_path": str(self.weights_path),
            "weights_exist": self.weights_path.exists(),
            "model_loaded": self.model is not None,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device
        }

