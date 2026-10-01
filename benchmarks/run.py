from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

from app.db import Database
from app.decisions import DecisionProvider, JevDecisionProvider, RuleDecisionProvider
from app.retrieval import FastEmbedProvider, HybridRetriever

ROOT = Path(__file__).parent


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil(percentile_value / 100 * len(values)) - 1)
    return sorted(values)[index]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_benchmark(
    corpus_path: Path = ROOT / "corpus.json",
    queries_path: Path = ROOT / "queries.json",
    limit: int = 5,
    strategy: str = "fts",
    decision_provider: DecisionProvider | None = None,
) -> dict[str, Any]:
    if strategy not in {"fts", "dense", "hybrid"}:
        raise ValueError("strategy must be 'fts', 'dense', or 'hybrid'")
    corpus = load_json(corpus_path)
    queries = load_json(queries_path)
    with tempfile.TemporaryDirectory(prefix="contextvault-benchmark-") as directory:
        db = Database(Path(directory) / "benchmark.db")
        for document in corpus:
            db.create_file(**document)
        if strategy == "fts":
            hybrid_retriever = None
        elif strategy == "dense":
            hybrid_retriever = HybridRetriever(
                db,
                FastEmbedProvider(),
                semantic_weight=1.0,
                min_semantic_score=0.0,
            )
        elif decision_provider:
            hybrid_retriever = HybridRetriever(
                db,
                FastEmbedProvider(),
                min_semantic_score=0.0,
            )
        else:
            hybrid_retriever = HybridRetriever(db, FastEmbedProvider())

        rows = []
        for case in queries:
            if hybrid_retriever is not None:
                results = hybrid_retriever.search(
                    case["question"],
                    product=case.get("product"),
                    version=case.get("version"),
                    document_type=case.get("document_type"),
                    limit=limit,
                )
            else:
                results = db.search(
                    case["question"],
                    product=case.get("product"),
                    version=case.get("version"),
                    document_type=case.get("document_type"),
                    limit=limit,
                )
            expected = set(case["expected_files"])
            expected_outcome = case.get("expected_outcome") or (
                "answerable" if expected else "insufficient_evidence"
            )
            names = [result["name"] for result in results]
            hits = [name for name in names if name in expected]
            decision = None
            decision_error = None
            if decision_provider:
                try:
                    decision = asyncio.run(decision_provider.evaluate(case["question"], results))
                except Exception as exc:  # noqa: BLE001 - benchmarks must report provider failures, not hide them
                    decision_error = type(exc).__name__
            end_to_end_target = expected_outcome
            no_answer = expected_outcome == "insufficient_evidence"
            no_answer_correct = no_answer and not names
            first_hit_rank = next((index + 1 for index, name in enumerate(names) if name in expected), None)
            rows.append(
                {
                    "case_id": case.get("id"),
                    "question": case["question"],
                    "expected_files": sorted(expected),
                    "expected_outcome": expected_outcome,
                    "retrieved_files": names,
                    "hit": bool(hits),
                    "no_answer": no_answer,
                    "no_answer_correct": no_answer_correct,
                    "precision_at_k": (1.0 if no_answer_correct else 0.0) if no_answer else len(hits) / max(len(names), 1),
                    "recall_at_k": (1.0 if no_answer_correct else 0.0) if no_answer else len(set(hits)) / max(len(expected), 1),
                    "reciprocal_rank": 1 / first_hit_rank if first_hit_rank else 0.0,
                    "decision_provider": decision.provider if decision else None,
                    "decision_outcome": decision.outcome if decision else None,
                    "decision_confidence": decision.confidence if decision else None,
                    "decision_conflict_probability": decision.conflict_probability if decision else None,
                    "decision_latency_ms": decision.latency_ms if decision else None,
                    "decision_input_tokens": decision.input_tokens if decision else None,
                    "decision_cost_usd": decision.cost_usd if decision else None,
                    "decision_error": decision_error,
                    "end_to_end_outcome_correct": decision.outcome == end_to_end_target if decision else False,
                }
            )

    answerable_rows = [row for row in rows if row["expected_outcome"] == "answerable"]
    no_answer_rows = [row for row in rows if row["expected_outcome"] == "insufficient_evidence"]
    review_rows = [row for row in rows if row["expected_outcome"] == "review"]
    decision_latencies = [float(row["decision_latency_ms"]) for row in rows if row["decision_latency_ms"] is not None]
    decision_input_tokens = [int(row["decision_input_tokens"]) for row in rows if row["decision_input_tokens"] is not None]
    decision_costs = [float(row["decision_cost_usd"]) for row in rows if row["decision_cost_usd"] is not None]
    return {
        "strategy": strategy,
        "k": limit,
        "queries": len(rows),
        "answerable_queries": len(answerable_rows),
        "retrieval_hit_rate_answerable": mean(row["hit"] for row in answerable_rows) if answerable_rows else None,
        "retrieval_precision_at_k_answerable": (
            mean(row["precision_at_k"] for row in answerable_rows) if answerable_rows else None
        ),
        "retrieval_recall_at_k_answerable": (
            mean(row["recall_at_k"] for row in answerable_rows) if answerable_rows else None
        ),
        "retrieval_mrr_answerable": (
            mean(row["reciprocal_rank"] for row in answerable_rows) if answerable_rows else None
        ),
        "no_answer_queries": len(no_answer_rows),
        "no_answer_empty_result_rate": (
            mean(row["no_answer_correct"] for row in no_answer_rows) if no_answer_rows else None
        ),
        "review_queries": len(review_rows),
        "decision_providers_observed": (
            ",".join(sorted({row["decision_provider"] for row in rows if row["decision_provider"]}))
            if decision_provider
            else None
        ),
        "decision_configuration": (
            {
                "answerability_threshold": decision_provider.answerability_threshold,
                "conflict_threshold": decision_provider.conflict_threshold,
            }
            if isinstance(decision_provider, JevDecisionProvider)
            else None
        ),
        "decision_failure_rate": (
            mean(row["decision_error"] is not None for row in rows) if decision_provider else None
        ),
        "decision_latency_p50_ms": percentile(decision_latencies, 50),
        "decision_latency_p95_ms": percentile(decision_latencies, 95),
        "decision_usage_cases": len(decision_input_tokens),
        "decision_input_tokens_total": sum(decision_input_tokens) if decision_input_tokens else None,
        "decision_cost_usd_total": sum(decision_costs) if decision_costs else None,
        "end_to_end_outcome_accuracy": (
            mean(row["end_to_end_outcome_correct"] for row in rows)
            if decision_provider
            else None
        ),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ContextVault retrieval baseline benchmark")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus.json")
    parser.add_argument("--queries", type=Path, default=ROOT / "queries.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--strategy", choices=("fts", "dense", "hybrid"), default="fts")
    parser.add_argument("--decision-provider", choices=("none", "rules", "jev"), default="none")
    parser.add_argument("--answerability-threshold", type=float, default=0.7)
    parser.add_argument("--conflict-threshold", type=float, default=0.7)
    args = parser.parse_args()
    if args.decision_provider == "none":
        decision_provider = None
    elif args.decision_provider == "rules":
        decision_provider = RuleDecisionProvider()
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            parser.error("--decision-provider jev requires OPENROUTER_API_KEY")
        decision_provider = JevDecisionProvider(
            api_key,
            answerability_threshold=args.answerability_threshold,
            conflict_threshold=args.conflict_threshold,
        )
    print(
        json.dumps(
            run_benchmark(args.corpus, args.queries, args.limit, args.strategy, decision_provider),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
