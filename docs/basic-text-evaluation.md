# Basic-text retrieval diagnostic

This is a small seven-query, project-authored retrieval sanity check. It has six answerable queries and one no-answer query. It is not an independent test set and it does not evaluate a decision gate.

## Results at k=5

Answerable retrieval metrics are calculated only over the six answerable queries. The no-answer empty-result rate is calculated independently over the one no-answer query.

| Strategy | Answerable precision@5 | Answerable recall@5 | Answerable MRR | No-answer empty-result rate |
| --- | ---: | ---: | ---: | ---: |
| FTS lexical baseline | 0.833 | 0.833 | 0.833 | 1.000 |
| Dense-only FastEmbed | 0.917 | 1.000 | 1.000 | 0.000 |
| Strict hybrid | 0.833 | 0.833 | 0.833 | 1.000 |

## What this shows

Dense retrieval found all six intended source files and ranked them first in this small set, but it returned a candidate for the no-answer question. FTS and strict hybrid avoided that no-answer false positive, but each missed one answerable case. The values should not be averaged into one headline recall: they represent different trade-offs.

The later Jev evaluation is documented separately in the [robust-text decision-gate evaluation](robust-text-jev-evaluation.md). That evaluation tests a decision gate after retrieval; it does not change these retrieval-only findings.

## Reproduce

```bash
uv run python -m benchmarks.run \
  --corpus benchmarks/basic-text-corpus.json \
  --queries benchmarks/basic-text-queries.json \
  --strategy fts
```
