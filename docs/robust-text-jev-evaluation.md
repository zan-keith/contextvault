# Robust-text live Jev evaluation

This second diagnostic corpus contains 11 documents and 12 labelled cases. It is separate from the initial basic-text corpus and adds metadata traps, an explicitly superseded document, three no-answer cases, and a pair of contradictory active router procedures.

## Measured results at k=5

| Configuration | Candidate precision@5 | Candidate recall@5 | MRR | Raw no-answer accuracy | Jev decision accuracy | End-to-end outcome accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FTS lexical baseline | 0.917 | 0.917 | 0.750 | 0.667 | — | — |
| Dense-only retrieval | 0.667 | 0.833 | 0.750 | 0.333 | — | — |
| Broad hybrid candidates + live Jev | 0.667 | 0.833 | 0.750 | 0.333 | 1.0 | 1.0 |

## What Jev correctly handled

- Rejected unrelated web documents for a payment question constrained to the `web` product.
- Rejected unrelated candidate passages for a sourdough query.
- Rejected a retired gateway query because the only matching source was superseded.
- Returned `review` rather than `answerable` when two active edge-router guides gave incompatible instructions.
- Accepted all eight supported answerable cases.

## Interpretation

The retrieval layer still returns weak semantic candidates and needs scalable vector indexing and a stronger ranking stage. Jev materially improves the user-facing result because ContextVault withholds candidates unless Jev finds sufficient, non-conflicting evidence.

This does **not** prove universal 100% quality. The corpus is small, labelled by this project, and not independent. The next reliable milestone is a larger held-out set built from real anonymised documents or externally reviewed cases, with latency and cost captured alongside accuracy.

## Reproduce

```bash
uv run --env-file .env python -m benchmarks.run \
  --corpus benchmarks/robust-text-corpus.json \
  --queries benchmarks/robust-text-queries.json \
  --strategy hybrid \
  --decision-provider jev
```
