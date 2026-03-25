.PHONY: install run test lint format pre-commit docker-build docker-run docker-build-gpu docker-run-gpu

CUDA_TAG ?= 12.4.1-cudnn-runtime-ubuntu22.04
ONNXRT_GPU_VERSION ?= 1.24.4

install:
	uv sync

run:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest --cov=src --cov-report=term-missing

lint:
	uv run ruff check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

pre-commit:
	uv run pre-commit install

docker-build:
	docker build -t clip-web-service .

docker-run:
	docker run -p 8000:8000 clip-web-service

docker-build-gpu:
	docker build -f Dockerfile.gpu \
		--build-arg CUDA_TAG=$(CUDA_TAG) \
		--build-arg ONNXRT_GPU_VERSION=$(ONNXRT_GPU_VERSION) \
		-t clip-web-service:gpu .

docker-run-gpu:
	docker run --gpus all -p 8000:8000 clip-web-service:gpu
