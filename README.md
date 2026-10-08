# ContextVault

ContextVault is a metadata-aware engineering memory service. It stores technical knowledge with product, version, document-type, and lifecycle metadata, then retrieves focused evidence for a user question.

The first slice deliberately uses:

- FastAPI for the HTTP API
- SQLite FTS5 for a transparent retrieval baseline
- FastEmbed local embeddings for hybrid reranking
- deterministic metadata filters for product, version, and status
- paragraph-preserving chunks with source references
- tests and benchmarks that establish retrieval behaviour before adding a remote model provider

The intended architecture is hybrid retrieval plus a replaceable semantic decision provider. Jev is an optional answerability component, not the database or the primary retriever.

The opt-in `POST /answers` prototype adds a separate replaceable generation provider. It only generates after a trusted, non-degraded, conflict-free answerable decision, passes a path-free evidence manifest to the generator, and validates structured claim citations before returning an answer. Rules decisions and rules fallbacks are never treated as trusted generation approval; failures abstain closed.

## Local development

```bash
uv sync --extra test
make lint
make test
make serve
```

The default database is `contextvault.db`. Override it with `CONTEXTVAULT_DB`.

## Try the end-to-end demo

```bash
uv run python examples/local_demo.py
```

The demo creates a temporary local database, ingests a maintenance document through the HTTP API, and runs an FTS query. It performs no network call or permanent write.

## API

Start a local server with `make serve`, then visit [interactive docs](http://127.0.0.1:8000/docs), use the checked-in [OpenAPI schema](docs/openapi.json), or follow the copyable calls in [API examples](docs/api.md).

## Benchmarks
```bash
uv run python -m benchmarks.run --strategy fts
uv run python -m benchmarks.run --strategy dense
uv run python -m benchmarks.run --strategy hybrid

# Live end-to-end direct RAG vs Jev-gated answer diagnostic (requires API key)
OPENROUTER_API_KEY=... make benchmark-end-to-end

# Fully offline three-arm diagnostic (direct vs Jev-gated vs Jev-gated+recovery)
uv run python -m benchmarks.offline_diagnostic

# Verify recovery queries can reach every expected file in the corpus
uv run python -m benchmarks.recovery_audit

# Direct, fail-closed Jev benchmark: provider errors are recorded, never replaced with rules.
uv run --env-file .env python -m benchmarks.run \
  --corpus benchmarks/robust-text-corpus.json \
  --queries benchmarks/robust-text-queries.json \
  --strategy hybrid --decision-provider jev > /tmp/contextvault-robust-jev.json
uv run python -m benchmarks.thresholds /tmp/contextvault-robust-jev.json

# Offline bounded-concurrency ingestion/retrieval stress test; no Jev or network call.
uv run python -m benchmarks.stress --documents 200 --queries 500 --concurrency 8
```

See the [basic-text evaluation](docs/basic-text-evaluation.md) for the deliberately limited retrieval comparison. The [robust-text decision-gate evaluation](docs/robust-text-jev-evaluation.md) contains the current control matrix, raw per-case artifacts, measured live latency/cost, and its explicit limitations. The [end-to-end evaluation](docs/end-to-end-evaluation.md) compares direct RAG, Jev-gated RAG, and Jev-gated RAG with targeted recovery search.

The app uses the rules decision provider by default. Set `OPENROUTER_API_KEY` to enable the OpenRouter Jev Decisions adapter (`typesafe/jev-1.13`); it automatically falls back to rules if the external provider is unavailable. Never commit that key. `typesafe/jev-router` is reserved for a future answer-generation routing step.

The stress command uses a temporary SQLite database and deterministic fixtures. It reports ingestion and retrieval throughput, p50/p95/p99 latency, and errors. It is offline by default and does not exercise Jev; use the robust-text command above for an explicitly live Jev evaluation.

For the larger labelled retrieval benchmark, see the [BEIR SciFact evaluator](docs/beir-scifact-evaluation.md). It reads temporary JSONL files, so the scientific corpus is not vendored in Git.

## Container

```bash
docker compose up --build
```

See `docs/deployment.md` for configuration, security boundaries, and the production hardening list.

## Current API slice

- `GET /health`
- `POST /files`
- `GET /files`
- `POST /queries` (backward-compatible evidence and decision response)
- `POST /answers` (opt-in, cited-answer prototype)
- `POST /answers/recovery` (opt-in, cited-answer prototype with targeted recovery search)

The current ingestion endpoint accepts extracted text as JSON or uploads UTF-8 `.txt`, `.md`, `.csv`, `.json`, and text-layer `.pdf` files. Uploaded bytes are retained in a local content-addressed store and linked from the file record. Scanned PDFs use local Tesseract OCR when `tesseract-ocr` is installed; otherwise the API returns a clear install message. Authentication, tenant isolation, and background workers are deliberately out of scope for the open-source technical core at this stage.

Answer generation is disabled without `OPENROUTER_API_KEY`; `OPENROUTER_GENERATION_MODEL` selects the OpenRouter chat-completions model. The answer slice intentionally has no persistence, streaming, authentication/tenant isolation, or vector database yet. Evidence is untrusted data, not instructions, and malformed/provider-error answers return a structured abstention.

`POST /answers/recovery` adds the recovery loop. It re-asks a rejected question with its salient terms, searches the same metadata-filtered database for the missing fact, merges only new chunks, and re-runs the Jev gate before abstaining (bounded by `max_rounds`, default 2). Generation is still gated on a trusted, non-degraded, conflict-free answerable decision, and the generator still only sees the server-built evidence manifest. Every round is recorded in the response for auditability.
