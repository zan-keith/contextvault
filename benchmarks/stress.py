"""Small, reproducible, offline concurrency benchmark for ContextVault."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import tempfile
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.db import Database


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(percentile / 100 * len(values)) - 1)
    return round(sorted(values)[index], 3)


async def _measure_phase(
    operations: int,
    concurrency: int,
    operation: Callable[[int], Awaitable[None]],
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors: Counter[str] = Counter()

    async def measured(index: int) -> None:
        async with semaphore:
            started = time.perf_counter()
            try:
                await operation(index)
            except Exception as exc:  # noqa: BLE001 - stress output must report failures
                errors[type(exc).__name__] += 1
            finally:
                latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(measured(index) for index in range(operations)))
    elapsed = time.perf_counter() - started
    return {
        "operations": operations,
        "errors": sum(errors.values()),
        "error_types": dict(sorted(errors.items())),
        "elapsed_seconds": round(elapsed, 6),
        "throughput_ops_per_second": round(operations / elapsed, 3) if elapsed else 0.0,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
        },
    }


async def run_stress(
    *, documents: int = 200, queries: int = 500, concurrency: int = 8, seed: int = 7
) -> dict[str, Any]:
    """Measure concurrent local ingestion and FTS retrieval in a temporary database."""
    if documents < 1 or queries < 1 or concurrency < 1:
        raise ValueError("documents, queries, and concurrency must be positive")

    randomizer = random.Random(seed)
    topics = ["calibration", "deployment", "vacuum", "routing", "certificate"]
    document_topics = [randomizer.choice(topics) for _ in range(documents)]
    selected_topics = [randomizer.choice(document_topics) for _ in range(queries)]

    with tempfile.TemporaryDirectory(prefix="contextvault-stress-") as directory:
        database = Database(Path(directory) / "stress.db")

        async def ingest(index: int) -> None:
            topic = document_topics[index]
            await asyncio.to_thread(
                database.create_file,
                name=f"stress-{index}.txt",
                description=f"Offline {topic} procedure",
                content=f"The {topic} procedure for fixture {index} is documented here.",
                product="stress-fixture",
                version="v1",
                document_type="runbook",
                status="active",
            )

        ingestion = await _measure_phase(documents, concurrency, ingest)

        async def retrieve(index: int) -> None:
            topic = selected_topics[index]
            results = await asyncio.to_thread(
                database.search, f"How do I handle {topic}?", product="stress-fixture", limit=5
            )
            if not results:
                raise AssertionError(f"no fixture result for topic {topic}")

        retrieval = await _measure_phase(queries, concurrency, retrieve)

    return {
        "mode": "offline",
        "config": {
            "documents": documents,
            "queries": queries,
            "concurrency": concurrency,
            "seed": seed,
        },
        "phases": {"ingestion": ingestion, "retrieval": retrieval},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline ContextVault concurrency benchmark")
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                run_stress(
                    documents=args.documents,
                    queries=args.queries,
                    concurrency=args.concurrency,
                    seed=args.seed,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
