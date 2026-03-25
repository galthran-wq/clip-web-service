from fastapi import APIRouter

from src.api.endpoints import classify, health

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(classify.router, prefix="/classify", tags=["classify"])
