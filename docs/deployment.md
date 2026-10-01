# Deployment and operations

## Local container

```bash
docker compose up --build
curl http://localhost:8000/health
```

The SQLite database and uploaded originals live in the `contextvault-data` volume. Back up that volume before upgrades. The container uses the rules provider unless `TYPESAFE_API_KEY` is supplied through the environment.

## Configuration

- `CONTEXTVAULT_DB`: SQLite database path; default `contextvault.db` outside the container.
- `CONTEXTVAULT_STORAGE_DIR`: original-file storage directory.
- `TYPESAFE_API_KEY`: optional Jev credential. Never put it in `.env` committed to Git or in an image layer.

## Current security boundary

This is an MVP and does not yet implement user authentication, workspace isolation, malware scanning, encryption at rest, quotas, or signed download URLs. Do not deploy it with confidential documents until those controls are added.

The current API returns a local `source_uri` for development traceability. A hosted version should return a document ID and a controlled download endpoint instead of filesystem paths.

## Before production

1. Add authentication and workspace-scoped foreign keys.
2. Move originals to object storage and encrypt them.
3. Add upload size limits and malware scanning.
4. Add a queue for extraction and model calls.
5. Add rate limits, structured logs, metrics, backups, and migrations.
6. Move synchronous OCR into a separate worker with page and resource limits.
7. Expand the labelled retrieval benchmark before changing ranking logic.
