"""Dependency-light BEIR JSONL retrieval evaluator for ContextVault."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from app.db import Database
from app.retrieval import FastEmbedProvider, HybridRetriever


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(percentile / 100 * len(values)) - 1)
    return round(sorted(values)[index], 3)


def _rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss


def _dcg(relevances: list[float]) -> float:
    return sum((2**relevance - 1) / math.log2(rank + 2) for rank, relevance in enumerate(relevances))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr_at_10": 0.0, "ndcg_at_10": 0.0}
    recalls = {k: [] for k in (1, 5, 10)}
    mrrs: list[float] = []
    ndcgs: list[float] = []
    for row in rows:
        relevant = row["relevant"]
        retrieved = row["retrieved"]
        for k, values in recalls.items():
            values.append(
                sum(relevant.get(document_id, 0) > 0 for document_id in retrieved[:k])
                / len(relevant)
            )
        first_rank = next(
            (rank for rank, document_id in enumerate(retrieved[:10], 1) if relevant.get(document_id, 0) > 0),
            None,
        )
        mrrs.append(1 / first_rank if first_rank else 0.0)
        actual = [relevant.get(document_id, 0) for document_id in retrieved[:10]]
        ideal = sorted(relevant.values(), reverse=True)[:10]
        ideal_dcg = _dcg(ideal)
        ndcgs.append(_dcg(actual) / ideal_dcg if ideal_dcg else 0.0)
    return {
        "recall_at_1": sum(recalls[1]) / len(rows),
        "recall_at_5": sum(recalls[5]) / len(rows),
        "recall_at_10": sum(recalls[10]) / len(rows),
        "mrr_at_10": sum(mrrs) / len(rows),
        "ndcg_at_10": sum(ndcgs) / len(rows),
    }


def evaluate_beir(
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    *,
    strategy: str = "fts",
    max_queries: int | None = None,
    qrels_only: bool = True,
    candidate_limit: int = 5000,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Evaluate FTS, dense, or hybrid retrieval against BEIR qrels."""
    if strategy not in {"fts", "dense", "hybrid"}:
        raise ValueError("strategy must be 'fts', 'dense', or 'hybrid'")
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be positive")
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")

    corpus = _load_jsonl(corpus_path)
    queries = _load_jsonl(queries_path)
    qrel_rows = _load_jsonl(qrels_path)
    qrels: dict[str, dict[str, float]] = {}
    for qrel in qrel_rows:
        score = float(qrel["score"])
        if score > 0:
            qrels.setdefault(str(qrel["query_id"]), {})[str(qrel["corpus_id"])] = score
    selected_queries = [
        query for query in queries if not qrels_only or str(query["_id"]) in qrels
    ]
    if max_queries is not None:
        selected_queries = selected_queries[:max_queries]

    rss_before = _rss_bytes()
    started = time.perf_counter()
    query_rows: list[dict[str, Any]] = []
    errors: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="contextvault-beir-") as directory:
        database = Database(Path(directory) / "beir.db")
        database.create_files(
            [
                {
                    "name": str(document["_id"]),
                    "description": document.get("title", ""),
                    "content": f"{document.get('title', '')}\n{document.get('text', '')}".strip(),
                    "product": "beir-scifact",
                    "version": "1",
                    "document_type": "scientific-paper",
                    "status": "active",
                }
                for document in corpus
            ]
        )
        build_seconds = time.perf_counter() - started
        retriever = None
        if strategy != "fts":
            retriever = HybridRetriever(
                database,
                FastEmbedProvider(embedding_model or FastEmbedProvider.name),
                semantic_weight=1.0 if strategy == "dense" else 0.65,
                min_semantic_score=0.0 if strategy == "dense" else 0.65,
                candidate_limit=candidate_limit,
            )
        for query in selected_queries:
            query_started = time.perf_counter()
            error = None
            try:
                results = (
                    retriever.search(query["text"], limit=10)
                    if retriever is not None
                    else database.search(query["text"], limit=10)
                )
                retrieved = list(dict.fromkeys(str(result["name"]) for result in results))
            except Exception as exc:  # noqa: BLE001 - benchmark output must retain per-query failures
                retrieved = []
                error = type(exc).__name__
                errors[error] = errors.get(error, 0) + 1
            query_rows.append(
                {
                    "query_id": str(query["_id"]),
                    "relevant": qrels.get(str(query["_id"]), {}),
                    "retrieved": retrieved,
                    "error": error,
                    "latency_ms": (time.perf_counter() - query_started) * 1000,
                }
            )

    query_elapsed = sum(row["latency_ms"] for row in query_rows) / 1000
    total_seconds = time.perf_counter() - started
    rss_after = _rss_bytes()
    latencies = [row["latency_ms"] for row in query_rows]
    return {
        "benchmark": "beir-scifact",
        "config": {
            "strategy": strategy,
            "qrels_only": qrels_only,
            "max_queries": max_queries,
            "candidate_limit": candidate_limit,
            "embedding_model": getattr(getattr(retriever, "embedder", None), "name", None),
            "top_k": 10,
        },
        "counts": {
            "corpus": len(corpus),
            "queries": len(queries),
            "qrels": len(qrel_rows),
            "evaluated_queries": len(selected_queries),
        },
        "metrics": _metrics(query_rows),
        "performance": {
            "index_build_wall_seconds": round(build_seconds, 6),
            "index_build_documents_per_second": round(len(corpus) / build_seconds, 3)
            if build_seconds
            else 0.0,
            "query_latency_ms": {
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "p99": _percentile(latencies, 99),
            },
            "query_throughput_per_second": round(len(query_rows) / query_elapsed, 3)
            if query_elapsed
            else 0.0,
            "total_wall_seconds": round(total_seconds, 6),
            "memory": {
                "available": rss_after is not None,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "rss_delta_bytes": rss_after - rss_before
                if rss_before is not None and rss_after is not None
                else None,
            },
            "errors": sum(errors.values()),
            "error_types": errors,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ContextVault on BEIR SciFact JSONL files")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--strategy", choices=("fts", "dense", "hybrid"), default="fts")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--all-queries", action="store_true", help="Include queries without qrels")
    parser.add_argument("--candidate-limit", type=int, default=5000)
    parser.add_argument(
        "--embedding-model",
        help="FastEmbed model for dense/hybrid strategies (default: BAAI/bge-small-en-v1.5)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_beir(
                args.corpus,
                args.queries,
                args.qrels,
                strategy=args.strategy,
                max_queries=args.max_queries,
                qrels_only=not args.all_queries,
                candidate_limit=args.candidate_limit,
                embedding_model=args.embedding_model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
