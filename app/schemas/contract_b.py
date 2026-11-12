from typing import Literal, Optional
from pydantic import BaseModel, Field

class ContractBResponse(BaseModel):
    """
    Contract B — Cloud -> Pi (Section 5)
    Strict payload received and parsed by Member 1's fail-safe alert logic.
    
    Fields:
      event_id: Matches request event_id
      accident_confirmed: The verdict boolean
      confidence: Confidence score between 0.0 and 1.0
      status: "ok", "error", or "timeout"
    """
    event_id: str = Field(
        ...,
        description="Matches the request event_id so Member 1 knows which event this response is for"
    )
    accident_confirmed: bool = Field(
        ...,
        description="The accident detection verdict"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) from the model/inference engine"
    )
    status: Literal["ok", "error", "timeout"] = Field(
        ...,
        description="Execution status: 'ok', 'error', or 'timeout'. Helps Member 1 distinguish 'no accident' from 'something broke'"
    )

    # Optional internal diagnostic detail (will serialize if present or can be stripped)
    detail: Optional[str] = Field(
        None,
        description="Optional diagnostic/error message for debugging"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_20261111_001",
                "accident_confirmed": True,
                "confidence": 0.94,
                "status": "ok"
            }
        }

    @classmethod
    def ok(cls, event_id: str, accident_confirmed: bool, confidence: float, detail: Optional[str] = None):
        return cls(
            event_id=event_id,
            accident_confirmed=accident_confirmed,
            confidence=round(float(confidence), 4),
            status="ok",
            detail=detail
        )

    @classmethod
    def error(cls, event_id: str, detail: Optional[str] = "Internal processing error"):
        return cls(
            event_id=event_id,
            accident_confirmed=False,
            confidence=0.0,
            status="error",
            detail=detail
        )

    @classmethod
    def timeout_response(cls, event_id: str, detail: Optional[str] = "Inference exceeded timeout deadline"):
        return cls(
            event_id=event_id,
            accident_confirmed=False,
            confidence=0.0,
            status="timeout",
            detail=detail
        )

