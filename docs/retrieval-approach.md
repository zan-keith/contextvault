# Retrieval approach

## Current baseline

The first implementation uses SQLite FTS5 over paragraph-preserving chunks. Product, version, and lifecycle status are applied as deterministic filters before ranked text retrieval. The baseline is intentionally transparent so later changes can be measured against it.

Current supported inputs are UTF-8 `.txt`, `.md`, `.csv`, and `.json` uploads. PDF/OCR, embeddings, and external model calls are not silently faked; they are separate implementation slices.

## Why this problem is worth testing

RAG quality depends on retrieval quality. Corrective Retrieval Augmented Generation describes a retrieval evaluator that assesses retrieved-document quality and triggers corrective actions when retrieval is weak: https://arxiv.org/abs/2401.15884

Long-context models can still underuse relevant information when it is surrounded by distracting material. *Lost in the Middle* reports position-sensitive degradation in multi-document question answering: https://arxiv.org/abs/2307.03172

RAG evaluation should separate retrieval relevance, generation faithfulness, and final answer quality. Ragas proposes metrics for these dimensions: https://arxiv.org/abs/2309.15217

Hierarchical summaries are a valid later direction for long documents. RAPTOR constructs representations at multiple abstraction levels, but ContextVault will not add that complexity until the baseline exposes a need: https://arxiv.org/abs/2401.18059

## Retrieval strategies

- `fts`: SQLite FTS5 plus exact metadata filters. This is the transparent baseline.
- `hybrid`: the same metadata boundary plus FTS contribution and local FastEmbed cosine similarity. It uses `BAAI/bge-small-en-v1.5` on CPU and rejects semantic-only candidates below the configured 0.65 similarity gate.

FastEmbed downloads the public embedding model on its first hybrid request. Query vectors are computed per request; document vectors are persisted in SQLite under the embedding model cache key and reused for later queries. If model initialisation or inference fails, ContextVault returns the FTS candidates with `retrieval_strategy: "fts_fallback"` instead of falsely reporting semantic results.


```bash
uv run python -m benchmarks.run --strategy fts
uv run python -m benchmarks.run --strategy dense
uv run python -m benchmarks.run --strategy hybrid

# Basic-text diagnostic corpus
uv run python -m benchmarks.run \
  --corpus benchmarks/basic-text-corpus.json \
  --queries benchmarks/basic-text-queries.json \
  --strategy hybrid
```

The checked-in baseline corpus and queries are intentionally small smoke data, not evidence of production accuracy. The latest local results are stored in `docs/baseline-results.json` and `docs/hybrid-baseline-results.json`; replace the corpus with a larger labelled set before drawing conclusions about embeddings or Jev.

## Planned comparison

1. Full-text search only.
2. Full-text search plus metadata filters.
3. Hybrid sparse/dense retrieval plus metadata filters.
4. Hybrid retrieval plus a Jev relevance/answerability provider.

Each version should be evaluated on labelled questions with known relevant files/passages. The project should report Precision@k, Recall@k, MRR or nDCG, citation precision, unsupported-answer rate, p50/p95 latency, and cost per query.

## Jev boundary

Jev is a decision provider, not the file database, primary retriever, or text generator. The planned interface is:

```python
class DecisionProvider(Protocol):
    async def evaluate(self, state, questions) -> DecisionResult: ...
```

The first production-like fallback will be rules or a mock provider. Jev-specific performance must be measured rather than assumed, because the project needs to remain testable when the external service is unavailable or changes.
