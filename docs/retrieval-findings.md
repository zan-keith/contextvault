# Retrieval and Jev: current findings

## What now works

ContextVault uses two different retrieval paths:

1. **Raw search** (`GET /files`) keeps a conservative semantic threshold so ordinary document search does not fill with weak matches.
2. **Question answering** (`POST /queries` with `strategy=hybrid`) sends a broader candidate set to Jev. Jev returns `answerable` only when the candidates support the question. If it returns `insufficient_evidence` or `review`, the API returns no visible sources.

This solves the observed trade-off in the initial corpus: the TLS paraphrase reaches Jev even though it has a score below the old retrieval threshold, while unrelated sourdough candidates are rejected before the user sees them.

## What the benchmark now shows

On the seven-case diagnostic corpus, hybrid retrieval plus Jev correctly handled all six answerable questions and the no-answer question end-to-end. Raw retrieval still returns weak candidates for a no-answer query; this is expected because the Jev step, rather than a fragile global score threshold, is the evidence gate.

## What still holds the project back

- **The dataset is tiny and project-authored.** Seven examples can reveal bugs but cannot establish production quality.
- **No independent holdout set exists.** Thresholds and design choices need testing against cases not used during implementation.
- **No conflict benchmark exists.** Jev's `review` path needs current versus superseded and genuinely incompatible source cases.
- **Retrieval is not scalable yet.** FastEmbed currently embeds every metadata-eligible chunk during a query; there is no persisted vector index or background indexing.
- **There is no answer-writing model yet.** Jev verifies evidence; it does not produce a user-facing prose answer. The next layer needs citation-preserving generation.
- **No latency/cost telemetry exists.** OpenRouter is an external billed dependency, so production needs timing, error, and usage metrics before broad rollout.

## Next evidence to collect

1. Build a larger held-out set with paraphrases, wrong-product distractors, stale documents, conflicts, and no-answer questions.
2. Benchmark retrieval, Jev evidence decisions, end-to-end outcomes, latency, and external-call failure fallback separately.
3. Persist embeddings and use vector indexing once the corpus is large enough that per-query embedding is no longer acceptable.
4. Add a cited answer layer only after the evidence gate remains reliable on held-out data.
