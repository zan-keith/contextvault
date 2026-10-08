.PHONY: setup test lint benchmark benchmark-dense benchmark-hybrid benchmark-jev benchmark-robust-jev benchmark-end-to-end benchmark-offline benchmark-thresholds benchmark-beir stress serve openapi

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

benchmark-jev:
	uv run --env-file .env python -m benchmarks.run --strategy hybrid --decision-provider jev

benchmark-robust-jev:
	uv run --env-file .env python -m benchmarks.run --corpus benchmarks/robust-text-corpus.json --queries benchmarks/robust-text-queries.json --strategy hybrid --decision-provider jev

benchmark-end-to-end:
	@test -n "$(OPENROUTER_API_KEY)" || (echo "Usage: OPENROUTER_API_KEY=... make benchmark-end-to-end" && exit 2)
	uv run python -m benchmarks.end_to_end --live --api-key "$(OPENROUTER_API_KEY)"

benchmark-offline:
	uv run python -m benchmarks.offline_diagnostic

benchmark-thresholds:
	@test -n "$(INPUT)" || (echo "Usage: make benchmark-thresholds INPUT=/path/to/benchmark.json" && exit 2)
	uv run python -m benchmarks.thresholds $(INPUT)

benchmark-beir:
	@test -n "$(DIR)" || (echo "Usage: make benchmark-beir DIR=/tmp/contextvault-beir/prepared [STRATEGY=fts] [MAX_QUERIES=]" && exit 2)
	uv run python -m benchmarks.beir --corpus $(DIR)/corpus.jsonl --queries $(DIR)/queries.jsonl --qrels $(DIR)/qrels.jsonl --strategy $(or $(STRATEGY),fts) $(if $(MAX_QUERIES),--max-queries $(MAX_QUERIES),)

stress:
	uv run python -m benchmarks.stress --documents 200 --queries 500 --concurrency 8

serve:
	@if [ -f .env ]; then uv run --env-file .env uvicorn app.main:app --reload; else uv run uvicorn app.main:app --reload; fi

openapi:
	uv run python scripts/export_openapi.py
