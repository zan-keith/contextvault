# Retrieval approach

## Current baseline

The first implementation uses SQLite FTS5 over paragraph-preserving chunks. Product, version, and lifecycle status are applied as deterministic filters before ranked text retrieval. The baseline is intentionally transparent so later changes can be measured against it.

Current supported inputs are UTF-8 `.txt`, `.md`, `.csv`, and `.json` uploads. PDF/OCR, embeddings, and external model calls are not silently faked; they are separate implementation slices.

## Why this problem is worth testing

RAG quality depends on retrieval quality. Corrective Retrieval Augmented Generation describes a retrieval evaluator that assesses retrieved-document quality and triggers corrective actions when retrieval is weak: https://arxiv.org/abs/2401.15884

Long-context models can still underuse relevant information when it is surrounded by distracting material. *Lost in the Middle* reports position-sensitive degradation in multi-document question answering: https://arxiv.org/abs/2307.03172

RAG evaluation should separate retrieval relevance, generation faithfulness, and final answer quality. Ragas proposes metrics for these dimensions: https://arxiv.org/abs/2309.15217

Hierarchical summaries are a valid later direction for long documents. RAPTOR constructs representations at multiple abstraction levels, but ContextVault will not add that complexity until the baseline exposes a need: https://arxiv.org/abs/2401.18059

## Run the benchmark

```bash
uv run python -m benchmarks.run
```

The checked-in baseline corpus and queries are intentionally small smoke data, not evidence of production accuracy. The latest local result is stored in `docs/baseline-results.json`; replace it with a larger labelled set before drawing conclusions about embeddings or Jev.

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
