import os
from pathlib import Path
from typing import Literal

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        ENVIRONMENT: str = "development"
        HOST: str = "0.0.0.0"
        PORT: int = 8000

        # Inference Mode: 'stub' (Phase 1) or 'model' (Phase 2)
        INFERENCE_MODE: Literal["stub", "model"] = "stub"

        # Timeout handling for Pi client (Section 5 / Section 7)
        TIMEOUT_SECONDS: float = Field(default=5.0, description="Max allowed execution seconds before timeout")

        # Confidence threshold for confirming an accident
        CONFIDENCE_THRESHOLD: float = 0.65

        # Phase 1 stub settings
        MOCK_INFERENCE_LATENCY: float = 0.35
        MOCK_DEFAULT_VERDICT: str = "random"

        # Phase 2 model settings
        MODEL_TYPE: str = "yolo"
        MODEL_WEIGHTS_PATH: str = "models/accident_detector.pt"
        MODEL_DEVICE: str = "cpu"

        # Video storage
        UPLOAD_DIR: str = "temp_uploads"
        CLEANUP_TEMP_FILES: bool = True
        MAX_UPLOAD_SIZE_MB: int = 50

        # Logging
        LOG_LEVEL: str = "INFO"
        LOG_FILE: str = "logs/cloud_inference.log"

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

    settings = Settings()

except ImportError:
    # Graceful fallback without pydantic_settings
    class SimpleSettings:
        def __init__(self):
            self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
            self.HOST = os.getenv("HOST", "0.0.0.0")
            self.PORT = int(os.getenv("PORT", "8000"))
            self.INFERENCE_MODE = os.getenv("INFERENCE_MODE", "stub")
            self.TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "5.0"))
            self.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
            self.MOCK_INFERENCE_LATENCY = float(os.getenv("MOCK_INFERENCE_LATENCY", "0.35"))
            self.MOCK_DEFAULT_VERDICT = os.getenv("MOCK_DEFAULT_VERDICT", "random")
            self.MODEL_TYPE = os.getenv("MODEL_TYPE", "yolo")
            self.MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "models/accident_detector.pt")
            self.MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
            self.UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp_uploads")
            self.CLEANUP_TEMP_FILES = os.getenv("CLEANUP_TEMP_FILES", "true").lower() in ("true", "1", "yes")
            self.MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
            self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
            self.LOG_FILE = os.getenv("LOG_FILE", "logs/cloud_inference.log")

    settings = SimpleSettings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(settings.LOG_FILE) or "logs").mkdir(parents=True, exist_ok=True)

