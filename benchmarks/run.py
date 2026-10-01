from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

from app.db import Database
from app.decisions import DecisionProvider, RuleDecisionProvider, provider_from_environment
from app.retrieval import FastEmbedProvider, HybridRetriever

ROOT = Path(__file__).parent


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
            names = [result["name"] for result in results]
            hits = [name for name in names if name in expected]
            decision = asyncio.run(decision_provider.evaluate(case["question"], results)) if decision_provider else None
            labeled_outcome = case.get("expected_outcome")
            evidence_target = labeled_outcome or ("answerable" if hits else "insufficient_evidence")
            end_to_end_target = labeled_outcome or ("answerable" if expected else "insufficient_evidence")
            no_answer = not expected
            no_answer_correct = no_answer and not names
            first_hit_rank = next((index + 1 for index, name in enumerate(names) if name in expected), None)
            rows.append(
                {
                    "question": case["question"],
                    "expected_files": sorted(expected),
                    "retrieved_files": names,
                    "hit": bool(hits),
                    "no_answer": no_answer,
                    "no_answer_correct": no_answer_correct,
                    "precision_at_k": (1.0 if no_answer_correct else 0.0) if no_answer else len(hits) / max(len(names), 1),
                    "recall_at_k": (1.0 if no_answer_correct else 0.0) if no_answer else len(set(hits)) / max(len(expected), 1),
                    "reciprocal_rank": 1 / first_hit_rank if first_hit_rank else 0.0,
                    "decision_provider": decision.provider if decision else None,
                    "decision_outcome": decision.outcome if decision else None,
                    "decision_correct_against_evidence": decision.outcome == evidence_target if decision else None,
                    "end_to_end_decision_correct": decision.outcome == end_to_end_target if decision else None,
                }
            )

    return {
        "strategy": strategy,
        "k": limit,
        "queries": len(rows),
        "hit_rate": mean(row["hit"] for row in rows) if rows else 0.0,
        "precision_at_k": mean(row["precision_at_k"] for row in rows) if rows else 0.0,
        "recall_at_k": mean(row["recall_at_k"] for row in rows) if rows else 0.0,
        "mrr": mean(row["reciprocal_rank"] for row in rows) if rows else 0.0,
        "no_answer_accuracy": mean(
            row["no_answer_correct"] for row in rows if row["no_answer"]
        ) if any(row["no_answer"] for row in rows) else None,
        "decision_provider": (
            ",".join(sorted({row["decision_provider"] for row in rows if row["decision_provider"]}))
            if decision_provider
            else None
        ),
        "decision_accuracy_against_evidence": (
            mean(row["decision_correct_against_evidence"] for row in rows)
            if decision_provider
            else None
        ),
        "end_to_end_decision_accuracy": (
            mean(row["end_to_end_decision_correct"] for row in rows)
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
    args = parser.parse_args()
    if args.decision_provider == "none":
        decision_provider = None
    elif args.decision_provider == "rules":
        decision_provider = RuleDecisionProvider()
    else:
        decision_provider = provider_from_environment()
        if isinstance(decision_provider, RuleDecisionProvider):
            parser.error("--decision-provider jev requires OPENROUTER_API_KEY")
    print(
        json.dumps(
            run_benchmark(args.corpus, args.queries, args.limit, args.strategy, decision_provider),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
