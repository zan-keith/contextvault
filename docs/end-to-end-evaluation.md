# End-to-end RAG evaluation

Run the live comparison with:

```sh
OPENROUTER_API_KEY=... make benchmark-end-to-end
```

The benchmark performs FTS retrieval once per query and gives the exact same
candidate list to every arm and the same `OpenRouterGenerationProvider`:

- `direct`: traditional RAG generation whenever candidates exist;
- `jev_gated`: direct generation only when the explicitly constructed Jev
  decision is trusted, answerable, non-degraded, and conflict-free;
- `jev_gated_recovery`: the same gate, plus a targeted recovery search that
  re-asks the question with its salient terms after a rejection, adds any new
  candidates, and re-runs the gate before abstaining.

The command requires `--live` through the Make target and fails closed without
`OPENROUTER_API_KEY`; it never uses the application provider fallback. Unit
tests inject deterministic providers and make no network calls.

The output includes action correctness, supported and unsupported answer rates,
abstentions, provider failures, p50/p95/p99 latency for retrieval, decision,
generation, and total stages, token/cost totals, and raw per-case records.
An answer is supported only for an `answerable` gold case whose citations all
resolve to an expected file. Answers on `insufficient_evidence` or `review`
cases, and citations to unexpected files, are unsupported. Jev is a gate, not
the answer-quality grader: the corpus/query gold outcome and expected-file
labels are the scoring basis.

The checked-in 30-case robust-text set is diagnostic only. It is intentionally
small and must not be treated as final proof of production RAG quality.

## Recovery loop

`POST /answers/recovery` and the `jev_gated_recovery` benchmark arm share
`app/recovery.py:answer_with_recovery`. After a trusted gate rejects the
initial candidates, the loop:

1. builds focused recovery queries from the question's salient terms
   (stopwords removed, duplicates dropped);
2. searches the same metadata-filtered database for each recovery query;
3. merges only new chunks into the candidate set (bounded by `limit`);
4. re-runs the decision gate on the merged set, up to `max_rounds` times.

Generation still only happens on a trusted, non-degraded, conflict-free
`answerable` decision, and the generator still only ever sees the server-built
evidence manifest. Conflicting evidence, untrusted or degraded decisions,
provider errors, and malformed answers all fail closed. Every round is
recorded in the response (`rounds`, `recovered_new_candidates`, per-round
decision outcome) so the caller can audit why an answer was or was not
produced.

## First live diagnostic run (16 cases, pre-recovery)

The first live run used the same FTS candidate set and the same OpenRouter
answer generator in both arms. It completed 16 cases and recorded the raw
artifact in `docs/end-to-end-diagnostic-results.json`.

- Direct RAG action correctness: **43.75%**.
- Jev-gated action correctness: **68.75%**.
- Direct unsupported-answer rate: **25%**.
- Jev-gated unsupported-answer rate: **0%**.
- Direct arm: 7 generated answers, 9 abstentions, 7 provider/format failures.
- Jev-gated arm: 3 generated answers, 13 abstentions, 4 provider/format failures.
- Jev-gated total cost: **$0.000622548** in this run.
- Direct total cost: **$0.0006318** in this run.
- Jev decision latency: p50 **302.55 ms**, p95 **551.78 ms**.
- Jev-gated total latency: p50 **314.14 ms**, p95 **2,942.35 ms**.

This was encouraging evidence that the gate prevented unsupported answers on
conflict/incomplete cases, but it was not a quality victory yet: the test was
small, multiple answer-generation calls failed, and the gated arm answered
fewer cases. The recovery loop addresses the coverage gap: instead of
abstaining when Jev says "not enough evidence," ContextVault now searches for
the missing fact and re-runs the gate.

## Offline three-arm result (30 cases, deterministic providers)

`benchmarks/offline_diagnostic.py` runs all three arms fully offline against
the 30-case robust-text corpus with a deterministic coverage gate (a stand-in
for Jev that approves only when every expected file is present) and an
extractive generator. Raw artifact: `docs/end-to-end-offline-diagnostic-results.json`.

- Direct RAG action correctness: **70.0%** (28 answers, 2 abstentions). It
  answers 9 cases whose gold outcome is `insufficient_evidence` or `review`.
- Jev-gated action correctness: **100%** (19 answers, 11 abstentions).
- Jev-gated + recovery action correctness: **100%** (19 answers, 11
  abstentions).

`benchmarks/recovery_audit.py` verifies that, for every one of the 30 cases,
the recovery queries can reach every expected file (artifact:
`docs/recovery-search-audit.json`). On this corpus the initial top-5 FTS
retrieval already surfaces the expected files, so recovery changes no outcomes
here; its value shows up when the first retrieval misses a fact (see the
multi-hop unit test in `tests/test_recovery.py`, where recovery adds the
missing second document and the gate then approves).

`tests/test_end_to_end.py` additionally runs all three arms offline with a
fake Jev and fake generator, verifying that both arms share the same FTS
candidates, that the gated arm never generates on a `review` decision, and
that the recovery arm matches or exceeds gated coverage.

The offline result verifies the plumbing (shared candidates, gating, recovery
merge, citation validation, per-arm telemetry) but not answer quality. The
next benchmark must use a larger held-out set and separate provider/format
failures from answer-quality failures.
