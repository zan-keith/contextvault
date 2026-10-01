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
