import base64
import binascii
import io

import structlog
from onnx_clip import OnnxClip, get_similarity_scores, softmax
from PIL import Image

logger = structlog.get_logger()


class CLIPService:
    def __init__(self, batch_size: int = 16) -> None:
        logger.info("clip_loading_model", batch_size=batch_size)
        self._model = OnnxClip(batch_size=batch_size)
        logger.info("clip_model_loaded")

    def decode_image(self, image_b64: str) -> Image.Image:
        try:
            image_bytes = base64.b64decode(image_b64)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (binascii.Error, OSError) as exc:
            raise ValueError(f"Invalid base64 image: {exc}") from exc

    def classify(self, images: list[Image.Image], prompts: list[str]) -> list[list[float]]:
        image_embeddings = self._model.get_image_embeddings(images)
        text_embeddings = self._model.get_text_embeddings(prompts)
        logits = get_similarity_scores(image_embeddings, text_embeddings)
        probs = softmax(logits)
        return [row.tolist() for row in probs]
