from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from src.dependencies import get_clip_service, set_clip_service
from src.main import app


@pytest.fixture
def mock_clip_service() -> MagicMock:
    mock = MagicMock()
    mock.classify.return_value = [[0.87, 0.13]]
    mock.decode_image.return_value = MagicMock()
    return mock


@pytest.fixture
async def client(mock_clip_service: MagicMock) -> AsyncIterator[AsyncClient]:
    set_clip_service(mock_clip_service)
    app.dependency_overrides[get_clip_service] = lambda: mock_clip_service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
    set_clip_service(None)
