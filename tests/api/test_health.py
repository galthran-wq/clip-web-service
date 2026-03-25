from httpx import ASGITransport, AsyncClient
from src.dependencies import set_clip_service
from src.main import app


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready(client: AsyncClient) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_returns_503_when_model_not_loaded() -> None:
    set_clip_service(None)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/ready")
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"
    finally:
        set_clip_service(None)
