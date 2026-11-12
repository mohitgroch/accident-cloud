import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.core.config import settings

def setup_logger(name: str = "accident_detection_cloud") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Avoid duplicate handlers if re-initialized
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] [evt=%(event_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not initialize file logger: {e}", file=sys.stderr)

    return logger

# Global logger instance
base_logger = setup_logger()

class EventLoggerAdapter(logging.LoggerAdapter):
    """Adapter to easily pass event_id into log records."""
    def process(self, msg, kwargs):
        event_id = kwargs.pop("event_id", self.extra.get("event_id", "-") if self.extra else "-")
        extra = kwargs.get("extra", {})
        extra["event_id"] = event_id
        kwargs["extra"] = extra
        return msg, kwargs

def get_logger(event_id: str = "-") -> EventLoggerAdapter:
    return EventLoggerAdapter(base_logger, {"event_id": event_id})

