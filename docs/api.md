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

The response contains ranked evidence and a decision object. The decision provider is local rules by default; when `OPENROUTER_API_KEY` is set, ContextVault calls OpenRouter's Jev Decisions API using `typesafe/jev-1.13` and falls back to rules if the external call fails. Hybrid question queries send a broad semantic candidate set to the decision provider; evidence is returned only for the `answerable` outcome.

## Generate a cited answer (prototype)

`POST /answers` is opt-in and does not change `POST /queries`. It retrieves evidence, asks the decision provider for an explicit trusted approval, and generates only when the decision is `answerable`, trusted, non-degraded, and conflict-free.

```bash
curl -X POST http://127.0.0.1:8000/answers \
  -H 'content-type: application/json' \
  -d '{"question":"What should I inspect after a vacuum alarm?","strategy":"fts"}'
```

An answer contains structured claims and `citation_ids`. The server validates every citation against a server-built manifest containing stable evidence IDs, source hashes, chunk/file locators, and text. Filesystem paths are never sent as citations. Successful responses also include generation telemetry (`provider`, model revision, latency, token counts, and provider-reported cost) when available. Insufficient, stale, conflicting, scope-mismatched, invalid, degraded, or untrusted decisions return `answer: null` with an `abstention` reason and action; the generator is not called.

Answer generation is disabled unless `OPENROUTER_API_KEY` is set, and the explicit adapter uses `OPENROUTER_GENERATION_MODEL` (default `openai/gpt-4o-mini`). Tests should inject a deterministic `GenerationProvider`; they never call OpenRouter. This is a prototype: there is no answer persistence, streaming, authentication/tenant isolation, or vector database, and evidence text remains untrusted data that the generator must not follow as instructions.

## Generate a cited answer with recovery

`POST /answers/recovery` adds a targeted recovery loop on top of the same gate. When the trusted decision rejects the initial evidence, the endpoint re-asks the question with its salient terms, searches the same metadata-filtered database for the missing fact, merges only new chunks, and re-runs the gate before abstaining.

```bash
curl -X POST http://127.0.0.1:8000/answers/recovery \
  -H 'content-type: application/json' \
  -d '{"question":"How many times should the system retry a failed card capture before abandoning it?","product":"finance","version":"v2","strategy":"fts","max_rounds":2}'
```

`max_rounds` (1-5, default 2) bounds the gate/recovery rounds. The response contains the final answer or a structured abstention, plus a `rounds` audit trail (per-round candidate names, decision outcome, latency, and how many new candidates recovery added). The same fail-closed rules as `POST /answers` apply: generation only happens on a trusted, non-degraded, conflict-free answerable decision, the generator only sees the server-built evidence manifest, and conflicting evidence, untrusted/degraded decisions, provider errors, and malformed answers all abstain.
