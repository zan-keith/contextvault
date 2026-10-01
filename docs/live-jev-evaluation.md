# Live Jev decision evaluation

This evaluation uses the seven-case basic-text diagnostic corpus and a live OpenRouter Jev Decisions call (`typesafe/jev-1.13`) for every question. It measures two layers independently:

- **Retrieval**: whether the expected source file was returned in the candidate set.
- **Decision**: whether Jev correctly judged the candidates as sufficient or insufficient.
- **Visible response**: whether the API would return evidence only for an answerable question.

## Results

| Configuration | Candidate recall@5 | Raw retrieval no-answer accuracy | Decision accuracy against candidates | End-to-end answer/abstain accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dense retrieval, no decision judge | 0.857 | 0.0 | — | — |
| Dense retrieval + Jev | 0.857 | 0.0 | 1.0 | 1.0 |
| Strict hybrid retrieval + local rules | 0.857 | 1.0 | 1.0 | 0.857 |
| Broad hybrid candidates + Jev | 0.857 | 0.0 | 1.0 | 1.0 |

The broader hybrid candidate path recovered the TLS paraphrase. It also returned unrelated candidates for the sourdough no-answer query, but Jev returned `insufficient_evidence`; the query API suppresses those candidates from the user-facing response.

## Honest conclusion

This small project-authored diagnostic dataset supports one narrow claim: Jev improves **evidence verification and safe abstention** compared with returning vector-search candidates directly. It does not prove that Jev improves document retrieval, nor does it establish production quality. See [retrieval findings](retrieval-findings.md) for the remaining limitations and next evaluation work.

## Reproduce

```bash
uv run --env-file .env python -m benchmarks.run \
  --corpus benchmarks/basic-text-corpus.json \
  --queries benchmarks/basic-text-queries.json \
  --strategy hybrid \
  --decision-provider jev
```
