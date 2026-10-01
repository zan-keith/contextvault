# ContextVault API

Start the server:

```bash
uv run uvicorn app.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`. The versioned OpenAPI document is checked in at `docs/openapi.json`.

## Health

```bash
curl http://127.0.0.1:8000/health
```

## Create an extracted-text record

```bash
curl -X POST http://127.0.0.1:8000/files \
  -H 'content-type: application/json' \
  -d '{
    "name": "x200-maintenance.txt",
    "description": "X200 vacuum maintenance procedure",
    "content": "Inspect the vacuum sensor and seal before retrying calibration.",
    "product": "X200",
    "version": "B",
    "document_type": "troubleshooting",
    "status": "active"
  }'
```

## Upload a document

```bash
curl -X POST http://127.0.0.1:8000/files/upload \
  -F 'file=@manual.pdf;type=application/pdf' \
  -F 'description=X200 calibration manual' \
  -F 'product=X200' \
  -F 'version=B' \
  -F 'document_type=troubleshooting' \
  -F 'status=active'
```

## Retrieve evidence

```bash
curl -G http://127.0.0.1:8000/files \
  --data-urlencode 'q=What should I inspect after X200 calibration fails?' \
  --data-urlencode 'product=X200' \
  --data-urlencode 'version=B' \
  --data-urlencode 'strategy=hybrid'
```

`strategy=fts` uses SQLite FTS5 only. `strategy=hybrid` combines FTS rank with local FastEmbed similarity and may return `retrieval_strategy: "fts_fallback"` if embeddings are unavailable.

## Query with an evidence decision

```bash
curl -X POST http://127.0.0.1:8000/queries \
  -H 'content-type: application/json' \
  -d '{
    "question": "What should I inspect after X200 calibration fails?",
    "product": "X200",
    "version": "B",
    "strategy": "hybrid"
  }'
```

The response contains ranked evidence and a decision object. The decision provider is local rules by default; when `TYPESAFE_API_KEY` is set, ContextVault uses the Jev adapter and falls back to rules if the external call fails.
