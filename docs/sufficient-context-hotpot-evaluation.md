# Sufficient Context reproduction benchmark

## Purpose

This benchmark tests the evidence-decision layer: whether the retrieved context contains enough information to answer a multi-hop question. It compares ContextVault's local rules judge with Jev.

## Dataset and construction

The source is the official HotPotQA distractor validation split mirrored at [Hugging Face](https://huggingface.co/datasets/hotpotqa/hotpot_qa). We selected 250 questions with at least two supporting Wikipedia passages and created 500 cases:

- 250 sufficient cases: all required supporting passages.
- 250 insufficient cases: one required supporting passage removed.

This is a reproducible proxy for the Sufficient Context benchmark. The original authors' 115 human-labelled examples are not published in their repository, so these numbers are not the official 115-example score.

## Results after the Jev prompt correction

| Judge | Accuracy | False-sufficient rate | Failures | p50 latency | p95 latency | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Local rules | 52.2% | 95.6% | 0/500 | 0.042 ms | 0.084 ms | $0 |
| Jev | 84.2% | 16.8% | 0/500 | 342 ms | 509 ms | $0.01465 / 500 |

The false-sufficient rate is the important safety measure here: it counts incomplete contexts that the judge incorrectly accepted as sufficient.

## Before/after Jev prompt correction

The old Jev instruction asked whether *any* passage supported an answer. The corrected instruction asks whether the *complete set* of passages contains all facts and multi-hop links needed for a definitive answer.

- Jev accuracy: 81.8% -> 84.2%.
- Jev false-sufficient rate: 30.8% -> 16.8%.
- Jev provider failures: 0 in both runs.

## Interpretation

Jev is substantially better than the local rules baseline at rejecting relevant-but-incomplete evidence. It is not a replacement for retrieval and this benchmark does not measure final generated-answer quality. Jev adds approximately 342 ms median latency and a small per-query cost, so it should be used when evidence quality matters and the rules judge can remain as an offline or outage fallback.
