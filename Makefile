.PHONY: setup test lint benchmark benchmark-hybrid serve openapi

setup:
	uv sync --extra test

test:
	uv run pytest

lint:
	uv run ruff check app benchmarks tests scripts examples

benchmark:
	uv run python -m benchmarks.run --strategy fts

benchmark-dense:
	uv run python -m benchmarks.run --strategy dense

benchmark-hybrid:
	uv run python -m benchmarks.run --strategy hybrid

serve:
	@if [ -f .env ]; then uv run --env-file .env uvicorn app.main:app --reload; else uv run uvicorn app.main:app --reload; fi

openapi:
	uv run python scripts/export_openapi.py
