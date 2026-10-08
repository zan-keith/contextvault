import json
from pathlib import Path

from benchmarks.beir import evaluate_beir


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_beir_evaluator_reports_qrels_metrics_and_timing(tmp_path):
    corpus = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"_id": "doc-a", "title": "Alpha", "text": "Alpha calibration procedure."},
            {"_id": "doc-b", "title": "Beta", "text": "Beta deployment procedure."},
        ],
    )
    queries = _write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {"_id": "q-a", "text": "alpha calibration"},
            {"_id": "q-unlabelled", "text": "not evaluated"},
        ],
    )
    qrels = _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_id": "q-a", "corpus_id": "doc-a", "score": 1}],
    )

    result = evaluate_beir(corpus, queries, qrels, strategy="fts")

    assert result["counts"] == {
        "corpus": 2,
        "queries": 2,
        "qrels": 1,
        "evaluated_queries": 1,
    }
    assert result["metrics"]["recall_at_1"] == 1.0
    assert result["metrics"]["recall_at_5"] == 1.0
    assert result["metrics"]["recall_at_10"] == 1.0
    assert result["metrics"]["mrr_at_10"] == 1.0
    assert result["metrics"]["ndcg_at_10"] == 1.0
    assert result["performance"]["errors"] == 0
    assert result["performance"]["query_latency_ms"]["p99"] >= 0
    assert result["config"]["embedding_model"] is None


def test_beir_records_the_explicit_dense_embedding_model(tmp_path):
    corpus = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [{"_id": "doc-a", "title": "Alpha", "text": "Alpha calibration procedure."}],
    )
    queries = _write_jsonl(tmp_path / "queries.jsonl", [{"_id": "q-a", "text": "alpha calibration"}])
    qrels = _write_jsonl(tmp_path / "qrels.jsonl", [{"query_id": "q-a", "corpus_id": "doc-a", "score": 1}])

    result = evaluate_beir(
        corpus,
        queries,
        qrels,
        strategy="dense",
        embedding_model="BAAI/bge-base-en-v1.5",
    )

    assert result["config"]["embedding_model"] == "BAAI/bge-base-en-v1.5"


def test_beir_max_queries_is_deterministic_and_can_include_unlabelled_queries(tmp_path):
    corpus = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [{"_id": "doc-a", "title": "Alpha", "text": "alpha"}],
    )
    queries = _write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {"_id": "q-a", "text": "alpha"},
            {"_id": "q-b", "text": "alpha"},
        ],
    )
    qrels = _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_id": "q-a", "corpus_id": "doc-a", "score": 1}],
    )

    result = evaluate_beir(
        corpus,
        queries,
        qrels,
        strategy="fts",
        max_queries=1,
        qrels_only=False,
    )

    assert result["counts"]["evaluated_queries"] == 1
    assert result["config"]["qrels_only"] is False


def test_beir_rejects_invalid_max_queries():
    import pytest

    with pytest.raises(ValueError, match="max_queries"):
        evaluate_beir(Path("missing"), Path("missing"), Path("missing"), max_queries=0)
