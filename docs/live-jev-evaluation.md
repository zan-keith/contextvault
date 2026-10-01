# Live Jev decision evaluation

This evaluation uses the seven-case basic-text diagnostic corpus and a live OpenRouter Jev Decisions call (`typesafe/jev-1.13`) for every question. It measures two layers independently:

- **Retrieval**: whether the expected source file was returned.
- **Decision**: whether Jev correctly judged the returned evidence as sufficient or insufficient.

## Results

| Configuration | Retrieval recall@5 | Retrieval no-answer accuracy | Decision accuracy against retrieved evidence | End-to-end answer/abstain accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dense retrieval, no decision judge | 0.857 | 0.0 | — | — |
| Dense retrieval + Jev | 0.857 | 0.0 | 1.0 | 1.0 |
| Hybrid retrieval + local rules | 0.857 | 1.0 | 1.0 | 0.857 |
| Hybrid retrieval + Jev | 0.857 | 1.0 | 1.0 | 0.857 |

The dense retriever returned unrelated documents for the sourdough no-answer query. Jev correctly returned `insufficient_evidence` despite those candidates, which is the important practical gain: it can stop a downstream answer model from treating weak retrieval as evidence.

Hybrid retrieval already abstained for sourdough but missed the TLS paraphrase before Jev could evaluate it. That is a retrieval calibration issue, not a Jev failure.

## Honest conclusion

This is a small, project-authored diagnostic dataset, not a production claim. It provides evidence that Jev improves **evidence verification and answer abstention** over dense retrieval alone. It does not prove that Jev improves document retrieval. A larger held-out, independently labelled corpus is needed before making a broader quality claim.

## Reproduce

```bash
uv run --env-file .env python -m benchmarks.run \
  --corpus benchmarks/basic-text-corpus.json \
  --queries benchmarks/basic-text-queries.json \
  --strategy dense \
  --decision-provider jev
```
