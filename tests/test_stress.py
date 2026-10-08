import asyncio
import json

from benchmarks.stress import run_stress


def test_offline_stress_reports_bounded_phase_metrics():
    result = asyncio.run(run_stress(documents=4, queries=6, concurrency=2, seed=7))

    assert result["mode"] == "offline"
    assert result["config"] == {"documents": 4, "queries": 6, "concurrency": 2, "seed": 7}
    assert set(result["phases"]) == {"ingestion", "retrieval"}
    for phase in result["phases"].values():
        assert phase["operations"] > 0
        assert phase["errors"] == 0
        assert phase["throughput_ops_per_second"] > 0
        assert phase["latency_ms"]["p50"] >= 0
        assert phase["latency_ms"]["p95"] >= phase["latency_ms"]["p50"]
        assert phase["latency_ms"]["p99"] >= phase["latency_ms"]["p95"]

    json.dumps(result)
