# Local benchmark artifacts

This directory is deliberately tracked only as a container and policy:

- `.gitignore` ignores all generated artifacts by default;
- this `README.md` and `.gitignore` remain tracked;
- raw benchmark outputs can include retrieved document text, prompts, provider
  names, tokens, costs, hashes, local paths, and timestamps, so they must not
  be committed casually.

Write runs here, never to `docs/` or the repository root:

```sh
mkdir -p benchmark-results/beir-scifact
uv run python -m benchmarks.beir \
  --corpus /tmp/contextvault-beir/prepared/corpus.jsonl \
  --queries /tmp/contextvault-beir/prepared/queries.jsonl \
  --qrels /tmp/contextvault-beir/prepared/qrels.jsonl \
  --strategy fts \
  > benchmark-results/beir-scifact/contextvault-fts.json

uv run python -m benchmarks.beir \
  --corpus /tmp/contextvault-beir/prepared/corpus.jsonl \
  --queries /tmp/contextvault-beir/prepared/queries.jsonl \
  --qrels /tmp/contextvault-beir/prepared/qrels.jsonl \
  --strategy dense \
  --embedding-model BAAI/bge-base-en-v1.5 \
  > benchmark-results/beir-scifact/bge-base-dense.json
```

Before sharing a result externally, create a reviewed, redacted summary in
`docs/`. Do not force-add files from this directory unless their contents have
been intentionally reviewed for source text, credentials, local paths, and
provider-sensitive metadata.
