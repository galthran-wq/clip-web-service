# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install        # Install dependencies (uv sync)
make run            # Dev server with hot reload on :8000
make test           # pytest with coverage
make lint           # ruff check + mypy strict
make format         # ruff format + ruff check --fix
make docker-build   # Build Docker image
make docker-run     # Run container on :8000
```

Run a single test file: `uv run pytest tests/api/test_classify.py`
Run a single test: `uv run pytest tests/api/test_classify.py::test_classify_batch_success -v`

## Architecture

FastAPI microservice that exposes CLIP zero-shot classification via ONNX Runtime (`onnx_clip`). Layered architecture with `src/` as the root package.

**Layers:**
- `src/api/endpoints/` — Thin route handlers. One router per file, registered in `src/api/router.py`. Use sync `def` for CPU-bound inference (FastAPI runs in threadpool), `async def` for I/O-bound.
- `src/schemas/` — Pydantic request/response models.
- `src/services/` — Business logic. Must not import FastAPI; injected into endpoints via `src/dependencies.py` using `Depends()`.
- `src/core/` — Middleware (request ID, logging, CORS) and exception handling (`AppError`).

**Key patterns:**
- App factory: `create_app()` in `src/main.py` wires middleware, exception handlers, routers, and Prometheus.
- Lifespan: `CLIPService` is loaded at startup in the lifespan context manager and stored via `set_clip_service()` in `src/dependencies.py`.
- Configuration: `src/config.py` uses Pydantic Settings, loaded from env vars / `.env` file. Singleton `settings` instance.
- CLIP inference: `src/services/clip_service.py` wraps `onnx_clip.OnnxClip`. Handles base64 decoding, image/text embedding, cosine similarity, softmax. No torch dependency.
- Logging: `structlog` with JSON output in prod, console in dev (controlled by `DEBUG` env var). Use `structlog.get_logger()` and log with key-value pairs.
- Errors: Raise `AppError(status_code, detail)` from `src/core/exceptions.py` — never catch broad `Exception`.
- Request ID: Auto-generated UUID per request, bound to structlog context vars, returned in `x-request-id` header.
- Readiness: `/ready` returns 503 until the CLIP model is loaded, then 200. `/health` always returns 200 (liveness).

## API

```
POST /classify/batch
{
  "images": [{"image_b64": "<base64>"}],
  "prompts": ["a photo", "a drawing"]
}
→ {"results": [{"index": 0, "probabilities": [0.87, 0.13]}]}
```

- Max batch size: 64 images (configurable via `CLIP_MAX_BATCH_SIZE`)
- Prompts: minimum 2 required (softmax over prompts)
- Model: CLIP ViT-B/32 via ONNX Runtime (no GPU needed)

## Adding a New Endpoint

1. Create `src/api/endpoints/<domain>.py` with `router = APIRouter()`
2. Create request/response schemas in `src/schemas/<domain>.py`
3. Register router in `src/api/router.py` via `router.include_router()`
4. Add tests in `tests/api/test_<domain>.py` (mirror the endpoint file structure)

## Code Style

- Python 3.12+, all functions require type annotations (params and return)
- Line length: 120
- Use `async def` for I/O-bound functions, sync `def` for CPU-bound (inference)
- mypy strict mode with pydantic plugin
- Import order enforced by ruff: stdlib → third-party → local
- Use `collections.abc` for abstract types, built-in generics (`list[str]` not `List[str]`)

## Testing

- All tests are `async def` — pytest-asyncio in auto mode
- Use the `client` fixture from `tests/conftest.py` (httpx `AsyncClient` with `ASGITransport`)
- `mock_clip_service` fixture provides a `MagicMock` — no real model loading in tests
- Test file structure mirrors `src/api/endpoints/`

## Configuration

| Env var | Default | Description |
|---|---|---|
| `CLIP_BATCH_SIZE` | 16 | Internal ONNX batch size |
| `CLIP_MAX_BATCH_SIZE` | 64 | Max images per request |
| `DEBUG` | false | Console logging instead of JSON |
| `METRICS_ENABLED` | true | Prometheus /metrics endpoint |
