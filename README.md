# CLIP Web Service

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type_checker-mypy-blue)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

CLIP zero-shot image classification microservice. Accepts base64-encoded images and text prompts, returns softmax probabilities.

Uses [onnx_clip](https://github.com/lakeraai/onnx_clip) (ONNX Runtime) — no PyTorch dependency, fast CPU inference (~5-10ms/image).

## Quick Start

```bash
make install
make run
# Server starts on http://localhost:8000
```

## API

```bash
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": [{"image_b64": "'$(base64 -w0 photo.jpg)'"}],
    "prompts": ["a real photograph", "a cartoon drawing"]
  }'
```

Response:

```json
{
  "results": [
    {"index": 0, "probabilities": [0.92, 0.08]}
  ]
}
```

- **Max batch**: 64 images per request
- **Prompts**: minimum 2 (softmax over prompts)
- **Model**: CLIP ViT-B/32 (downloaded automatically on first run)

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /classify/batch` | Zero-shot classification |
| `GET /health` | Liveness probe (always 200) |
| `GET /ready` | Readiness probe (503 until model loaded) |
| `GET /metrics` | Prometheus metrics |

## Commands

| Command | Description |
|---|---|
| `make install` | Install dependencies |
| `make run` | Dev server with hot reload |
| `make test` | Run tests with coverage |
| `make lint` | ruff + mypy |
| `make format` | Auto-format code |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run container on :8000 |

## Configuration

| Env var | Default | Description |
|---|---|---|
| `CLIP_BATCH_SIZE` | 16 | Internal ONNX inference batch size |
| `CLIP_MAX_BATCH_SIZE` | 64 | Max images per request |
| `DEBUG` | false | Console logging instead of JSON |
| `METRICS_ENABLED` | true | Prometheus /metrics endpoint |

## Stack

- **FastAPI** + **uvicorn** — async web framework
- **onnx_clip** + **ONNX Runtime** — CLIP inference without PyTorch
- **uv** — package manager
- **Pydantic v2** — validation and settings
- **structlog** — structured logging
- **Prometheus** — metrics
- **pytest + httpx** — testing
- **ruff + mypy** — linting and type checking

## Project Structure

```
src/
├── main.py              — app factory, lifespan (model loading), logging
├── config.py            — pydantic-settings configuration
├── dependencies.py      — CLIPService dependency injection
├── api/
│   ├── router.py        — aggregated API router
│   └── endpoints/
│       ├── health.py    — /health, /ready
│       └── classify.py  — POST /classify/batch
├── schemas/
│   ├── health.py        — HealthResponse
│   └── classify.py      — ClassifyBatchRequest/Response
├── services/
│   └── clip_service.py  — CLIP inference via onnx_clip
└── core/
    ├── exceptions.py    — AppError + handlers
    └── middleware.py     — CORS, request logging, request ID
```
