from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

class BaseInferenceEngine(ABC):
    """
    Abstract interface for AI accident verification.
    Both the Phase 1 Stub engine and Phase 2 Member 3 Model engine
    implement this contract.
    """

    @abstractmethod
    async def predict(
        self,
        video_path: str,
        sensor_summary: Optional[Dict[str, Any]] = None,
        gps_info: Optional[Dict[str, Any]] = None,
        event_id: str = "unknown"
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Runs accident verification on the packaged video clip and telemetry.

        Returns:
            accident_confirmed (bool): True if collision confirmed, False otherwise.
            confidence (float): Float in range [0.0, 1.0].
            diagnostic (str, optional): Additional info or model label.
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns engine metadata (mode, model type, device, etc.)."""
        pass

