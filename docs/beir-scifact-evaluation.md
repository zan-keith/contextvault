# BEIR SciFact retrieval evaluation

`benchmarks.beir` evaluates ContextVault FTS, dense, or hybrid retrieval against BEIR SciFact qrels. It reports corpus/query/qrel counts, index build wall time and document throughput, optional RSS memory (when `psutil` is installed), Recall@1/5/10, MRR@10, nDCG@10, query p50/p95/p99 latency, query throughput, and errors. Queries without a positive qrel are excluded by default.

The raw dataset stays outside this repository. The following commands download the public BEIR release, prepare the test qrels under `/tmp`, and run the evaluator:

```bash
mkdir -p /tmp/contextvault-beir/raw /tmp/contextvault-beir/prepared
curl -L --fail https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
  -o /tmp/contextvault-beir/scifact.zip
unzip -q -o /tmp/contextvault-beir/scifact.zip -d /tmp/contextvault-beir/raw
cp /tmp/contextvault-beir/raw/scifact/corpus.jsonl /tmp/contextvault-beir/prepared/corpus.jsonl
cp /tmp/contextvault-beir/raw/scifact/queries.jsonl /tmp/contextvault-beir/prepared/queries.jsonl
cp /tmp/contextvault-beir/raw/scifact/qrels/test.tsv /tmp/contextvault-beir/prepared/qrels.tsv
awk 'BEGIN { FS=OFS="\t" } NR > 1 { print "{\"query_id\":\"" $1 "\",\"corpus_id\":\"" $2 "\",\"score\":" $3 "}" }' \
  /tmp/contextvault-beir/prepared/qrels.tsv > /tmp/contextvault-beir/prepared/qrels.jsonl

/home/keith/.hermes/bin/uv run python -m benchmarks.beir \
  --corpus /tmp/contextvault-beir/prepared/corpus.jsonl \
  --queries /tmp/contextvault-beir/prepared/queries.jsonl \
  --qrels /tmp/contextvault-beir/prepared/qrels.jsonl \
  --strategy fts > /tmp/contextvault-beir/scifact-fts.json
```

The same evaluator supports `--strategy dense` and `--strategy hybrid`; select the tested embedding model explicitly with `--embedding-model` (for example, `BAAI/bge-base-en-v1.5`). This makes ContextVault compete against a materially stronger external dense retriever rather than only its default small embedding model. For a bounded comparison, add `--max-queries`; a full run should evaluate all qrel-bearing queries at the same `top_k=10` and report the exact model, corpus checksum, platform, latency, and nDCG/MRR/Recall metrics. `make benchmark-beir DIR=/tmp/contextvault-beir/prepared STRATEGY=fts` remains a convenience wrapper.

The raw result artifacts are deliberately not checked in. Write their JSON output under `benchmark-results/` (ignored by Git); preserve only reviewed, redacted summaries in this document. The historical result files are local-only and must not be force-added without review.

SciFact is a scientific claim-verification collection with 5,183 abstracts and expert-written claims; this benchmark measures document retrieval only and does not evaluate claim classification or rationale extraction. See the [BEIR dataset listing](https://github.com/beir-cellar/beir/wiki/Datasets-available), the [BEIR repository](https://github.com/beir-cellar/beir), and Wadden et al., [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974).
