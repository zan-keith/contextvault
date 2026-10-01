import json
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
    assert result["retrieval_precision_at_k_answerable"] == 1.0
    assert result["retrieval_recall_at_k_answerable"] == 1.0
    assert result["no_answer_empty_result_rate"] == 1.0


def test_benchmark_reports_answerable_retrieval_metrics_separately_from_no_answer_rejection():
    result = run_benchmark(
        Path("benchmarks/corpus.json"),
        Path("benchmarks/queries.json"),
        limit=5,
    )

    assert result["answerable_queries"] == 4
    assert result["retrieval_recall_at_k_answerable"] == 1.0
    assert result["retrieval_mrr_answerable"] == 1.0
    assert result["no_answer_queries"] == 1
    assert result["no_answer_empty_result_rate"] == 1.0
    assert "recall_at_k" not in result


def test_threshold_summary_reports_accuracy_coverage_and_calibration():
    from benchmarks.thresholds import summarize_thresholds

    rows = [
        {"expected_outcome": "answerable", "decision_confidence": 0.8, "decision_conflict_probability": 0.1},
        {"expected_outcome": "insufficient_evidence", "decision_confidence": 0.6, "decision_conflict_probability": 0.1},
        {"expected_outcome": "review", "decision_confidence": 0.9, "decision_conflict_probability": 0.85},
    ]

    summaries = summarize_thresholds(rows, answerability_thresholds=[0.5, 0.7], conflict_threshold=0.7)

    assert summaries[0]["answerability_threshold"] == 0.5
    assert summaries[0]["end_to_end_outcome_accuracy"] == 2 / 3
    assert summaries[0]["false_answer_rate"] == 0.5
    assert summaries[1]["answerability_threshold"] == 0.7
    assert summaries[1]["end_to_end_outcome_accuracy"] == 1.0
    assert summaries[1]["answerable_precision"] == 1.0
    assert summaries[1]["brier_score"] == 1.21 / 3


def test_benchmark_records_decision_failures_without_substituting_a_fallback():
    class BrokenJudge:
        async def evaluate(self, _question, _candidates):
            raise RuntimeError("network unavailable")

    result = run_benchmark(
        Path("benchmarks/corpus.json"),
        Path("benchmarks/queries.json"),
        decision_provider=BrokenJudge(),
    )

    assert result["decision_failure_rate"] == 1.0
    assert result["end_to_end_outcome_accuracy"] == 0.0
    assert {case["decision_error"] for case in result["cases"]} == {"RuntimeError"}


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

    assert result["decision_providers_observed"] == "evidence-judge"
    assert result["end_to_end_outcome_accuracy"] == 1.0


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


def test_benchmark_honours_a_labeled_review_outcome(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    queries_path = tmp_path / "queries.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "name": "router-a.txt",
                    "description": "Edge router incident guidance",
                    "content": "Restart the edge router after collecting logs.",
                    "product": "edge",
                    "version": "v1",
                    "document_type": "runbook",
                    "status": "active",
                },
                {
                    "name": "router-b.txt",
                    "description": "Edge router incident guidance",
                    "content": "Never restart the edge router; fail traffic to standby.",
                    "product": "edge",
                    "version": "v1",
                    "document_type": "runbook",
                    "status": "active",
                },
            ]
        )
    )
    queries_path.write_text(
        json.dumps(
            [
                {
                    "question": "What should we do when the edge router fails?",
                    "expected_files": ["router-a.txt", "router-b.txt"],
                    "expected_outcome": "review",
                }
            ]
        )
    )

    class ReviewJudge:
        async def evaluate(self, _question, _candidates):
            return DecisionResult("review", 1.0, ["conflict"], "review-judge")

    result = run_benchmark(corpus_path, queries_path, decision_provider=ReviewJudge())

    assert result["end_to_end_outcome_accuracy"] == 1.0
