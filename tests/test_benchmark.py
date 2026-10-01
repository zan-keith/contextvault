from pathlib import Path

from benchmarks.run import run_benchmark


def test_checked_in_baseline_benchmark():
    result = run_benchmark(
        Path("benchmarks/corpus.json"),
        Path("benchmarks/queries.json"),
        limit=5,
    )

    assert result["queries"] == 5
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
    assert result["no_answer_accuracy"] == 1.0


def test_benchmark_can_run_dense_only_comparison(monkeypatch):
    captured = {}

    class FakeRetriever:
        def __init__(self, database, _embedder, **kwargs):
            captured.update(kwargs)
            self.database = database

        def search(self, question, **filters):
            return self.database.search(question, **filters)

    monkeypatch.setattr("benchmarks.run.FastEmbedProvider", lambda: object())
    monkeypatch.setattr("benchmarks.run.HybridRetriever", FakeRetriever)

    result = run_benchmark(
        Path("benchmarks/corpus.json"),
        Path("benchmarks/queries.json"),
        limit=5,
        strategy="dense",
    )

    assert result["strategy"] == "dense"
    assert captured == {"semantic_weight": 1.0, "min_semantic_score": 0.0}
