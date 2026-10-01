from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

from app.db import Database
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
) -> dict[str, Any]:
    if strategy not in {"fts", "hybrid"}:
        raise ValueError("strategy must be 'fts' or 'hybrid'")
    corpus = load_json(corpus_path)
    queries = load_json(queries_path)
    with tempfile.TemporaryDirectory(prefix="contextvault-benchmark-") as directory:
        db = Database(Path(directory) / "benchmark.db")
        for document in corpus:
            db.create_file(**document)
        hybrid_retriever = HybridRetriever(db, FastEmbedProvider()) if strategy == "hybrid" else None

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
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ContextVault retrieval baseline benchmark")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus.json")
    parser.add_argument("--queries", type=Path, default=ROOT / "queries.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--strategy", choices=("fts", "hybrid"), default="fts")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.corpus, args.queries, args.limit, args.strategy), indent=2))


if __name__ == "__main__":
    main()
