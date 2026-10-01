# Architecture

```text
                    +---------------------+
                    | FastAPI HTTP API     |
                    | /files /queries      |
                    +----------+----------+
                               |
             +-----------------+-----------------+
             |                                   |
             v                                   v
   Document ingestion                    Evidence retrieval
   - retain original bytes               - metadata filters
   - extract text layer                  - SQLite FTS5 baseline
   - OCR scanned PDFs (optional)         - FastEmbed hybrid reranking
             |                                   |
             v                                   v
  SQLite files / chunks / FTS5           Decision provider
  local content-addressed storage        - Jev when configured
                                         - deterministic rules fallback
```

## Boundaries

- **SQLite and deterministic filters** own exact facts: active status, product, version, and document type.
- **FTS5** is the transparent lexical baseline.
- **FastEmbed** provides local semantic similarity for metadata-eligible chunks.
- **Jev** is an optional typed decision provider for answerability and conflict checks. It does not replace the retriever or the database.
- **Original bytes** are retained separately from extracted chunks so answers remain traceable to the source file.

## Intentional non-goals

ContextVault is currently an open-source technical core. It does not implement accounts, billing, multi-tenancy, a hosted frontend, or other SaaS business features.
