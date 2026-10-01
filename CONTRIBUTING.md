# Contributing to ContextVault

Thanks for helping improve ContextVault.

## Development setup

```bash
uv sync --extra test
uv run pytest
uv run python -m benchmarks.run
```

Use Python 3.11 or newer. The project uses local SQLite and does not require external API keys for its test suite.

## Before opening a pull request

1. Start with a failing behavioural test for a new feature or bug fix.
2. Keep file metadata and original-source retention intact; retrieval results must remain traceable to source content.
3. Run the full test suite and benchmark.
4. Update documentation when an API, configuration value, retrieval strategy, or safety boundary changes.
5. Do not commit secrets, local databases, uploaded documents, benchmark data containing private material, or model API keys.

## Design principles

- Retrieval and model decisions are separate components.
- Exact constraints such as workspace, status, product, and version belong in deterministic code and database queries.
- Model providers are optional adapters with a local fallback.
- Benchmarks must report measured results; do not present a toy corpus as production performance.
- New document formats must preserve original bytes and fail clearly when extraction is unavailable.

## Commit style

Keep commits focused and imperative, for example:

```text
feat: add document type filtering
fix: reject empty OCR output
chore: document local development setup
```
