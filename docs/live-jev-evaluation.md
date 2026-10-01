# Early seven-case Jev diagnostic (historical)

This document records the first live Jev experiment on the basic-text corpus. It is retained for development history, but its original summary used a blended retrieval metric and omitted the direct `dense + local rules` control. It must **not** be read as the current evidence for a Jev advantage.

The current evaluation uses:

- answerable-query retrieval recall/MRR separated from no-answer empty-result rate;
- explicit `answerable` / `insufficient_evidence` / `review` labels for every query;
- `dense + rules` and `hybrid + rules` controls;
- fail-closed Jev benchmark mode, with provider errors retained per case;
- per-case answerability/conflict probabilities and observed provider;
- latency, returned usage/cost, and offline threshold-sweep artifacts.

See the [robust-text decision-gate evaluation](robust-text-jev-evaluation.md) for the current 16-case diagnostic matrix, raw result files, and limitations.

## Historical observation

The initial run indicated that a Jev gate could reject an obviously unrelated sourdough candidate returned by dense retrieval. That was only a sanity check. It did not establish superiority over a local rule gate, calibrated decision thresholds, conflict handling, or real-document performance.

## Original reproduction command

```bash
uv run --env-file .env python -m benchmarks.run \
  --corpus benchmarks/basic-text-corpus.json \
  --queries benchmarks/basic-text-queries.json \
  --strategy hybrid \
  --decision-provider jev
```
