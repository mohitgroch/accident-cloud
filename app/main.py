import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import get_logger
from app.api.endpoints import router as api_router
from app.schemas.contract_b import ContractBResponse

logger = get_logger("SYSTEM")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("=" * 60)
    logger.info("AI-Assisted Smart Vehicle Accident Detection Cloud Service")
    logger.info(f"Environment    : {settings.ENVIRONMENT}")
    logger.info(f"Inference Mode : {settings.INFERENCE_MODE.upper()}")
    logger.info(f"Timeout Budget : {settings.TIMEOUT_SECONDS}s")
    logger.info(f"Listening on   : http://{settings.HOST}:{settings.PORT}")
    logger.info("=" * 60)
    yield
    # Shutdown logic
    logger.info("Shutting down Accident Detection Cloud Service...")

app = FastAPI(
    title="AI-Assisted Smart Vehicle Accident Detection Cloud API",
    description=(
        "Cloud-AI verification backend for BMSCE Major Project Work. "
        "Receives trigger clips & telemetry from Raspberry Pi (Contract A) and returns "
        "verified collision verdicts (Contract B) with automated fail-safe timeout handling."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware (permits requests from Pi clients, local test harnesses, and web dashboards)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing & Diagnostic Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(process_time)
        return response
    except Exception as exc:
        process_time = round((time.time() - start_time) * 1000, 2)
        logger.exception(f"Unhandled server error on {request.url.path}: {exc}")
        
        # If the request was hitting the accident detection endpoint, fail safe with Contract B format!
        if "/detect" in request.url.path:
            return JSONResponse(
                status_code=500,
                content={
                    "event_id": "unknown_error",
                    "accident_confirmed": False,
                    "confidence": 0.0,
                    "status": "error",
                    "detail": f"Internal server exception: {str(exc)}"
                },
                headers={"X-Process-Time-Ms": str(process_time)}
            )
        raise exc

# Include API Router
app.include_router(api_router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "AI-Assisted Smart Vehicle Accident Detection and Emergency Alert System",
        "institution": "B.M.S. College of Engineering (Dept. of ECE)",
        "role": "Team Member 4 — Cloud & AI Deployment",
        "service_status": "online",
        "endpoints": {
            "detect": "POST /api/v1/detect (Contract A -> Contract B)",
            "health": "GET /health or /api/v1/health",
            "status": "GET /api/v1/status",
            "recent_events": "GET /api/v1/events/recent",
            "docs": "/docs (Swagger UI interactive testing)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

