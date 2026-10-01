from pathlib import Path

from app.decisions import DecisionResult
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


def test_benchmark_reports_decision_accuracy_for_retrieved_evidence():
    class EvidenceJudge:
        provider_name = "evidence-judge"

        async def evaluate(self, _question, candidates):
            return DecisionResult(
                outcome="answerable" if candidates else "insufficient_evidence",
                confidence=1.0,
                reasons=["deterministic test judge"],
                provider=self.provider_name,
            )

    result = run_benchmark(
        Path("benchmarks/corpus.json"),
        Path("benchmarks/queries.json"),
        limit=5,
        decision_provider=EvidenceJudge(),
    )

    assert result["decision_provider"] == "evidence-judge"
    assert result["decision_accuracy_against_evidence"] == 1.0
    assert result["end_to_end_decision_accuracy"] == 1.0


def test_benchmark_uses_broader_hybrid_candidates_when_a_decision_judge_is_present(monkeypatch):
    captured = {}

    class FakeRetriever:
        def __init__(self, database, _embedder, **kwargs):
            captured.update(kwargs)
            self.database = database

        def search(self, question, **filters):
            return self.database.search(question, **filters)

    class EvidenceJudge:
        async def evaluate(self, _question, candidates):
            return DecisionResult(
                outcome="answerable" if candidates else "insufficient_evidence",
                confidence=1.0,
                reasons=["deterministic test judge"],
                provider="evidence-judge",
            )

    monkeypatch.setattr("benchmarks.run.FastEmbedProvider", lambda: object())
    monkeypatch.setattr("benchmarks.run.HybridRetriever", FakeRetriever)

    run_benchmark(
        Path("benchmarks/corpus.json"),
        Path("benchmarks/queries.json"),
        strategy="hybrid",
        decision_provider=EvidenceJudge(),
    )

    assert captured == {"min_semantic_score": 0.0}
