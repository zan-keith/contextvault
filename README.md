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
```

See the [basic-text evaluation](docs/basic-text-evaluation.md) for the current, deliberately limited FTS/dense/hybrid comparison and its caveats.

The app uses the rules decision provider by default. Set `TYPESAFE_API_KEY` to enable the Jev adapter; it automatically falls back to rules if the external provider is unavailable. Never commit that key.

## Container

```bash
docker compose up --build
```

See `docs/deployment.md` for configuration, security boundaries, and the production hardening list.

## Current API slice

- `GET /health`
- `POST /files`
- `GET /files`
- `POST /queries`

The current ingestion endpoint accepts extracted text as JSON or uploads UTF-8 `.txt`, `.md`, `.csv`, `.json`, and text-layer `.pdf` files. Uploaded bytes are retained in a local content-addressed store and linked from the file record. Scanned PDFs use local Tesseract OCR when `tesseract-ocr` is installed; otherwise the API returns a clear install message. Authentication, tenant isolation, and background workers are deliberately out of scope for the open-source technical core at this stage.
