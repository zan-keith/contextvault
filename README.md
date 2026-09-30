# ContextVault

ContextVault is a metadata-aware engineering memory service. It stores technical knowledge with product, version, document-type, and lifecycle metadata, then retrieves focused evidence for a user question.

The first slice deliberately uses:

- FastAPI for the HTTP API
- SQLite FTS5 for a transparent retrieval baseline
- deterministic metadata filters for product, version, and status
- paragraph-preserving chunks with source references
- tests that establish retrieval behaviour before adding embeddings or Jev

The intended later architecture is hybrid retrieval plus a replaceable semantic decision provider. Jev is an optional reranking/answerability component, not the database or the primary retriever.

## Local development

```bash
uv venv
uv pip install -e '.[test]'
pytest
uvicorn app.main:app --reload
```

The default database is `contextvault.db`. Override it with `CONTEXTVAULT_DB`.

## Current API slice

- `GET /health`
- `POST /files`
- `GET /files`
- `POST /queries`

The current ingestion endpoint accepts extracted text as JSON or uploads UTF-8 `.txt`, `.md`, `.csv`, `.json`, and text-layer `.pdf` files. Uploaded bytes are retained in a local content-addressed store and linked from the file record. Scanned/image-only PDFs return a clear error until OCR is implemented. Embeddings, Jev integration, authentication, and background jobs are intentionally deferred until the baseline behaviour is measured.
