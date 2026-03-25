import base64
import io
from unittest.mock import MagicMock

from httpx import AsyncClient
from PIL import Image


def _make_test_image_b64() -> str:
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def test_classify_batch_success(client: AsyncClient, mock_clip_service: MagicMock) -> None:
    mock_clip_service.classify.return_value = [[0.87, 0.13]]
    response = await client.post(
        "/classify/batch",
        json={
            "images": [{"image_b64": _make_test_image_b64()}],
            "prompts": ["a photo", "a drawing"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["index"] == 0
    assert data["results"][0]["probabilities"] == [0.87, 0.13]
    mock_clip_service.classify.assert_called_once()


async def test_classify_batch_multiple_images(client: AsyncClient, mock_clip_service: MagicMock) -> None:
    mock_clip_service.classify.return_value = [[0.87, 0.13], [0.02, 0.98]]
    img_b64 = _make_test_image_b64()
    response = await client.post(
        "/classify/batch",
        json={
            "images": [{"image_b64": img_b64}, {"image_b64": img_b64}],
            "prompts": ["a photo", "a drawing"],
        },
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


async def test_classify_batch_exceeds_max_size(client: AsyncClient) -> None:
    img_b64 = _make_test_image_b64()
    response = await client.post(
        "/classify/batch",
        json={
            "images": [{"image_b64": img_b64}] * 65,
            "prompts": ["a photo", "a drawing"],
        },
    )
    assert response.status_code == 400
    assert "exceeds maximum" in response.json()["detail"]


async def test_classify_batch_invalid_base64(client: AsyncClient, mock_clip_service: MagicMock) -> None:
    mock_clip_service.decode_image.side_effect = ValueError("Invalid base64 image: bad data")
    response = await client.post(
        "/classify/batch",
        json={
            "images": [{"image_b64": "not_valid_base64!!!"}],
            "prompts": ["a photo", "a drawing"],
        },
    )
    assert response.status_code == 400
    assert "index 0" in response.json()["detail"]


async def test_classify_batch_empty_images(client: AsyncClient) -> None:
    response = await client.post(
        "/classify/batch",
        json={
            "images": [],
            "prompts": ["a photo", "a drawing"],
        },
    )
    assert response.status_code == 422


async def test_classify_batch_single_prompt(client: AsyncClient) -> None:
    response = await client.post(
        "/classify/batch",
        json={
            "images": [{"image_b64": _make_test_image_b64()}],
            "prompts": ["only one prompt"],
        },
    )
    assert response.status_code == 422


async def test_classify_batch_missing_prompts(client: AsyncClient) -> None:
    response = await client.post(
        "/classify/batch",
        json={
            "images": [{"image_b64": _make_test_image_b64()}],
        },
    )
    assert response.status_code == 422


async def test_classify_batch_inference_failure(client: AsyncClient, mock_clip_service: MagicMock) -> None:
    mock_clip_service.classify.side_effect = RuntimeError("ONNX session failed")
    response = await client.post(
        "/classify/batch",
        json={
            "images": [{"image_b64": _make_test_image_b64()}],
            "prompts": ["a photo", "a drawing"],
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Classification failed"
