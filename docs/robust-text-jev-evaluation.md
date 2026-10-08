# Robust-text decision-gate evaluation

This is a **project-authored diagnostic**, not a production-quality or independent held-out study. It contains 11 short documents and 16 labelled queries:

- 8 answerable queries;
- 7 insufficient-evidence queries, including four topical-but-incomplete questions (retry count, missing-parcel compensation, certificate authority, and MFA recovery time);
- 1 active-document conflict expected to return `review`.

Every query carries an explicit `expected_outcome`: `answerable`, `insufficient_evidence`, or `review`. The benchmark reports the retrieved filenames, ranks, decision probabilities, observed provider, error (if any), and final outcome per case. It does **not** call a retrieval-conditioned score an independently labelled evidence-quality metric.

## Retrieval results at k=5

Retrieval metrics are reported only over the 8 answerable queries. No-answer empty-result rate is reported independently across the 7 insufficient-evidence queries.

| Retrieval strategy | Answerable recall@5 | Answerable MRR | No-answer empty-result rate |
| --- | ---: | ---: | ---: |
| FTS | 1.000 | 1.000 | 0.286 |
| Dense | 1.000 | 1.000 | 0.143 |
| Hybrid | 1.000 | 1.000 | 0.286 |

This corpus remains retrieval-friendly: all supported documents appeared at rank 1 for all three strategies. It is useful for testing evidence decisions after imperfect candidate retrieval, but it does **not** demonstrate a retrieval-ranking advantage for dense or hybrid search.

## Decision-gate control matrix

| Retrieval | Judge | End-to-end outcome accuracy | Observed judge failures |
| --- | --- | ---: | ---: |
| FTS | none | — | — |
| Dense | none | — | — |
| Hybrid | none | — | — |
| Dense | local rules | 0.438 (7/16) | 0/16 |
| Dense | live Jev | 1.000 (16/16) | 0/16 |
| Hybrid | local rules | 0.438 (7/16) | 0/16 |
| Hybrid | live Jev | 1.000 (16/16) | 0/16 |

The `jev` benchmark mode is **fail closed**: a remote request error is retained as a failed case with `decision_error`; it never silently runs rules instead. The application may still use a rules fallback for service availability, but that is deliberately not used in this evaluation.

This comparison supports a narrow claim: on this exact 16-case diagnostic set, live Jev produced better labelled end-to-end outcomes than the checked-in local rule gate. It does not establish an advantage on independently labelled evidence sets, real company documents, or general production traffic.

## Live operational measurements

| Configuration | p50 decision latency | p95 decision latency | Input tokens | Reported total cost |
| --- | ---: | ---: | ---: | ---: |
| Dense + Jev | 280.61 ms | 904.80 ms | 8,933 | $0.000375186 |
| Hybrid + Jev | 289.72 ms | 378.36 ms | 8,947 | $0.000375774 |

These are one local live run against OpenRouter's Decisions API, not latency or cost service-level objectives.

## Threshold sweep and calibration

One fixed set of recorded Jev probabilities was reclassified offline at answerability thresholds `0.5`, `0.7`, `0.8`, and `0.9`, with conflict threshold fixed at `0.7`.

| Answerability threshold | End-to-end accuracy | Answerable precision | Answerable recall | False-answer rate | Abstention rate | Brier score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.438 | 0.0626 |
| 0.7 | 1.000 | 1.000 | 1.000 | 0.000 | 0.438 | 0.0626 |
| 0.8 | 1.000 | 1.000 | 1.000 | 0.000 | 0.438 | 0.0626 |
| 0.9 | 1.000 | 1.000 | 1.000 | 0.000 | 0.438 | 0.0626 |

The flat sweep is a limitation: all observed answerability probabilities were well away from these thresholds. This dataset therefore does **not** calibrate the production threshold. A future set needs independently labelled borderline evidence and more hard negatives around the decision boundary.

## Raw artifacts

Raw artifacts are generated locally under `benchmark-results/` and ignored by
Git because they can retain source passages, provider telemetry, and host
metadata. This document retains only the reviewed summary above. Re-run the
commands below to regenerate an artifact; do not force-add it without redaction
review.

## Reproduce

```bash
# A normal Jev benchmark: direct provider, error recorded per case, no fallback.
uv run --env-file .env python -m benchmarks.run \
  --corpus benchmarks/robust-text-corpus.json \
  --queries benchmarks/robust-text-queries.json \
  --strategy hybrid \
  --decision-provider jev > /tmp/robust-hybrid-jev.json

# Reclassify the recorded probabilities without calling Jev again.
uv run python -m benchmarks.thresholds /tmp/robust-hybrid-jev.json
```

## What remains necessary

1. An independently authored, held-out corpus with candidate-set-level human evidence judgements.
2. Borderline cases to calibrate answerability and conflict thresholds.
3. Multi-document, ambiguous, near-miss, malformed-metadata, and prompt-injection-like document cases.
4. Repeated latency, cost, and failure-rate runs under realistic corpus sizes.
