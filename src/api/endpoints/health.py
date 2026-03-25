from fastapi import APIRouter

from src.core.exceptions import AppError
from src.dependencies import get_clip_service
from src.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health")
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready")
async def readiness_check() -> HealthResponse:
    try:
        get_clip_service()
    except RuntimeError as exc:
        raise AppError(status_code=503, detail="Model not loaded") from exc
    return HealthResponse(status="ok")
