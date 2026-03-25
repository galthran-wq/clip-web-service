from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    image_b64: str = Field(..., max_length=20_000_000, description="Base64-encoded image (JPEG or PNG)")


class ClassifyBatchRequest(BaseModel):
    images: list[ImageInput] = Field(..., min_length=1, description="List of base64-encoded images")
    prompts: list[str] = Field(..., min_length=2, description="Text prompts for zero-shot classification")


class ClassifyResult(BaseModel):
    index: int
    probabilities: list[float]


class ClassifyBatchResponse(BaseModel):
    results: list[ClassifyResult]
