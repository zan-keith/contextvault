# Basic-text retrieval evaluation

This small, human-labelled diagnostic corpus contains seven ordinary operations and support documents. Its seven queries cover exact wording, four paraphrases, a metadata boundary, and a no-answer case. It is a design check, not a production-quality evaluation set.

## Measured results at k=5

| Strategy | Precision@5 | Recall@5 | MRR | No-answer accuracy |
| --- | ---: | ---: | ---: | ---: |
| FTS lexical baseline | 0.857 | 0.857 | 0.714 | 1.0 |
| Dense-only FastEmbed | 0.786 | 0.857 | 0.857 | 0.0 |
| Current hybrid | 0.857 | 0.857 | 0.714 | 1.0 |

Dense-only retrieval recovered the TLS paraphrase that FTS missed, but returned unrelated sources for the sourdough no-answer query. The current hybrid's conservative semantic threshold rejected both, so it tied FTS instead of improving it.

## Conclusion

This does not yet establish that ContextVault improves over ordinary dense RAG. It demonstrates the actual trade-off: dense retrieval improves paraphrase recall, while the current abstention gate improves no-answer precision but needs calibration on a larger held-out labelled corpus. Do not change the threshold based on this seven-query set.

Run the comparison:

```bash
for strategy in fts dense hybrid; do
  uv run python -m benchmarks.run \
    --corpus benchmarks/basic-text-corpus.json \
    --queries benchmarks/basic-text-queries.json \
    --strategy "$strategy"
done
```
