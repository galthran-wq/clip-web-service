import structlog
from fastapi import APIRouter, Depends

from src.config import settings
from src.core.exceptions import AppError
from src.dependencies import get_clip_service
from src.schemas.classify import ClassifyBatchRequest, ClassifyBatchResponse, ClassifyResult
from src.services.clip_service import CLIPService

logger = structlog.get_logger()

router = APIRouter()

_clip_dep = Depends(get_clip_service)


@router.post("/batch")
def classify_batch(
    body: ClassifyBatchRequest,
    clip_service: CLIPService = _clip_dep,  # noqa: B008
) -> ClassifyBatchResponse:
    if len(body.images) > settings.clip_max_batch_size:
        raise AppError(
            status_code=400,
            detail=f"Batch size {len(body.images)} exceeds maximum of {settings.clip_max_batch_size}",
        )

    images = []
    for i, img_input in enumerate(body.images):
        try:
            images.append(clip_service.decode_image(img_input.image_b64))
        except ValueError as exc:
            raise AppError(status_code=400, detail=f"Image at index {i}: {exc}") from exc

    logger.info("classify_batch", num_images=len(images), num_prompts=len(body.prompts))
    try:
        probabilities = clip_service.classify(images, body.prompts)
    except Exception as exc:
        logger.exception("classify_failed")
        raise AppError(status_code=500, detail="Classification failed") from exc

    results = [ClassifyResult(index=i, probabilities=probs) for i, probs in enumerate(probabilities)]
    return ClassifyBatchResponse(results=results)
