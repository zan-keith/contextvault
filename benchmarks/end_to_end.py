from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any

from app.answers import (
    GenerationProvider,
    OpenRouterGenerationProvider,
    build_evidence_manifest,
    validate_generated_answer,
)
from app.db import Database
from app.decisions import DecisionProvider, JevDecisionProvider
from app.recovery import DatabaseRecoverySearch, answer_with_recovery

ROOT = Path(__file__).parent


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(percentile_value / 100 * len(values)) - 1)]


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    return {f"{name}_ms": percentile(values, number) for name, number in (("p50", 50), ("p95", 95), ("p99", 99))}


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _direct_record(
    question: str,
    candidates: list[dict[str, Any]],
    expected_files: set[str],
    expected_outcome: str,
    generation_provider: GenerationProvider,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "action": "abstain",
        "generation_called": False,
        "provider_failure": None,
        "answer": None,
        "citations": [],
        "unsupported": False,
        "supported": False,
    }
    if not candidates:
        record["generation_latency_ms"] = 0.0
        return record
    record["generation_called"] = True
    try:
        generated = asyncio.run(generation_provider.generate(question, build_evidence_manifest(candidates)))
        answer = validate_generated_answer(generated, build_evidence_manifest(candidates))
        citation_ids = [citation_id for claim in answer.claims for citation_id in claim.citation_ids]
        evidence_by_id = {item["evidence_id"]: item for item in build_evidence_manifest(candidates)}
        cited_files = [evidence_by_id[citation_id]["name"] for citation_id in citation_ids]
        record.update(
            {
                "action": "answer",
                "answer": answer.model_dump(mode="json"),
                "citations": cited_files,
                "unsupported": expected_outcome != "answerable"
                or any(name not in expected_files for name in cited_files),
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "cost_usd": answer.cost_usd,
                "provider": answer.provider,
                "provider_version": answer.provider_version,
            }
        )
        record["supported"] = not record["unsupported"]
        record["generation_latency_ms"] = answer.latency_ms or 0.0
    except Exception as exc:  # noqa: BLE001 - benchmark reports failures per case
        record["provider_failure"] = type(exc).__name__
        record["generation_latency_ms"] = 0.0
    return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [record for record in records if record["action"] == "answer"]
    total_tokens = sum(
        (record.get("input_tokens") or 0)
        + (record.get("output_tokens") or 0)
        + (record.get("decision_input_tokens") or 0)
        for record in records
    )
    total_cost = sum(
        (record.get("cost_usd") or 0.0) + (record.get("decision_cost_usd") or 0.0)
        for record in records
    )
    action_correct = [
        record["action"] == ("answer" if record["expected_outcome"] == "answerable" else "abstain")
        for record in records
    ]
    summary = {
        "action_correctness": mean(action_correct) if action_correct else None,
        "supported_answer_rate": mean(record["supported"] for record in records) if records else None,
        "unsupported_answer_rate": mean(record["unsupported"] for record in records) if records else None,
        "abstentions": sum(record["action"] == "abstain" for record in records),
        "provider_failures": sum(record["provider_failure"] is not None for record in records),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "answers": len(answered),
    }
    for stage in ("retrieval", "decision", "generation", "total"):
        stage_summary = _latency_summary(
            [record[f"{stage}_latency_ms"] for record in records if record.get(f"{stage}_latency_ms") is not None]
        )
        summary[stage] = stage_summary
        summary.update({f"{stage}_latency_{key}": value for key, value in stage_summary.items()})
    return summary


def run_benchmark(
    corpus_path: Path = ROOT / "robust-text-corpus.json",
    queries_path: Path = ROOT / "robust-text-queries.json",
    *,
    generation_provider: GenerationProvider,
    decision_provider: DecisionProvider,
    limit: int = 5,
    max_rounds: int = 2,
    database_path: Path | None = None,
) -> dict[str, Any]:
    """Run three arms against one immutable candidate set per case.

    - ``direct``: generate whenever candidates exist.
    - ``jev_gated``: generate only on a trusted, non-degraded, conflict-free
      answerable decision.
    - ``jev_gated_recovery``: same gate, plus targeted recovery search that adds
      candidates and re-runs the gate before abstaining.

    Providers are dependencies so unit tests can remain completely offline. The
    CLI constructs the explicit OpenRouter and Jev adapters below.
    """
    corpus = _load(corpus_path)
    queries = _load(queries_path)
    temporary = tempfile.TemporaryDirectory(prefix="contextvault-e2e-") if database_path is None else None
    path = database_path or Path(temporary.name) / "benchmark.db"
    try:
        database = Database(path)
        database.create_files(corpus)
        recovery_search = DatabaseRecoverySearch(database)
        cases: list[dict[str, Any]] = []
        arm_records: dict[str, list[dict[str, Any]]] = {
            "direct": [],
            "jev_gated": [],
            "jev_gated_recovery": [],
        }
        for query in queries:
            expected_files = set(query.get("expected_files", []))
            expected_outcome = query.get("expected_outcome") or ("answerable" if expected_files else "insufficient_evidence")
            retrieval_started = time.perf_counter()
            candidates = database.search(
                query["question"],
                product=query.get("product"),
                version=query.get("version"),
                document_type=query.get("document_type"),
                limit=limit,
            )
            retrieval_latency = round((time.perf_counter() - retrieval_started) * 1000, 2)
            common = {
                "case_id": query.get("id"),
                "question": query["question"],
                "expected_files": sorted(expected_files),
                "expected_outcome": expected_outcome,
                "retrieved_files": [candidate["name"] for candidate in candidates],
            }

            direct = _direct_record(query["question"], candidates, expected_files, expected_outcome, generation_provider)
            direct.update(common, retrieval_latency_ms=retrieval_latency, decision_latency_ms=0.0)
            direct["total_latency_ms"] = round(retrieval_latency + direct["generation_latency_ms"], 2)

            decision_started = time.perf_counter()
            try:
                decision = asyncio.run(decision_provider.evaluate(query["question"], candidates))
                decision_error = None
            except Exception as exc:  # noqa: BLE001 - benchmark reports provider failures
                decision = None
                decision_error = type(exc).__name__
            decision_latency = round((time.perf_counter() - decision_started) * 1000, 2)
            non_conflicting = decision is not None and (
                decision.conflict_probability is None or decision.conflict_probability < 0.7
            )
            if decision is not None and decision.outcome == "answerable" and decision.trusted and not decision.degraded and non_conflicting:
                gated = _direct_record(query["question"], candidates, expected_files, expected_outcome, generation_provider)
            else:
                gated = {
                    "action": "abstain",
                    "generation_called": False,
                    "provider_failure": decision_error,
                    "answer": None,
                    "citations": [],
                    "unsupported": False,
                    "supported": False,
                    "generation_latency_ms": 0.0,
                }
            gated.update(
                common,
                retrieval_latency_ms=retrieval_latency,
                decision_latency_ms=decision_latency,
                decision_outcome=decision.outcome if decision else None,
                decision_provider=decision.provider if decision else None,
                decision_input_tokens=decision.input_tokens if decision else None,
                decision_cost_usd=decision.cost_usd if decision else None,
            )
            gated["total_latency_ms"] = round(retrieval_latency + decision_latency + gated["generation_latency_ms"], 2)

            recovery_started = time.perf_counter()
            recovery_result = asyncio.run(
                answer_with_recovery(
                    query["question"],
                    candidates,
                    decision_provider=decision_provider,
                    generation_provider=generation_provider,
                    recovery_search=recovery_search,
                    product=query.get("product"),
                    version=query.get("version"),
                    document_type=query.get("document_type"),
                    limit=limit,
                    max_rounds=max_rounds,
                )
            )
            recovery_latency = round((time.perf_counter() - recovery_started) * 1000, 2)
            if recovery_result["answer"] is not None:
                answer_payload = recovery_result["answer"]
                generation = recovery_result.get("generation") or {}
                citation_ids = [
                    citation_id for claim in answer_payload["claims"] for citation_id in claim["citation_ids"]
                ]
                evidence_by_id = {item["evidence_id"]: item for item in recovery_result["candidates"]}
                cited_files = [
                    evidence_by_id[citation_id]["name"]
                    for citation_id in citation_ids
                    if citation_id in evidence_by_id
                ]
                citation_is_supported = (
                    expected_outcome == "answerable"
                    and len(cited_files) == len(citation_ids)
                    and all(name in expected_files for name in cited_files)
                )
                recovered = {
                    "action": "answer",
                    "generation_called": True,
                    "provider_failure": None,
                    "answer": answer_payload,
                    "citations": cited_files,
                    "unsupported": not citation_is_supported,
                    "supported": citation_is_supported,
                    "generation_latency_ms": generation.get("latency_ms") or 0.0,
                    "input_tokens": generation.get("input_tokens"),
                    "output_tokens": generation.get("output_tokens"),
                    "cost_usd": generation.get("cost_usd"),
                    "provider": generation.get("provider"),
                    "provider_version": generation.get("provider_version"),
                }
            else:
                recovered = {
                    "action": "abstain",
                    "generation_called": False,
                    "provider_failure": None,
                    "answer": None,
                    "citations": [],
                    "unsupported": False,
                    "supported": False,
                    "generation_latency_ms": 0.0,
                }
            recovered.update(
                common,
                retrieval_latency_ms=retrieval_latency,
                decision_latency_ms=sum(
                    round_item["decision_latency_ms"] for round_item in recovery_result["rounds"]
                ),
                decision_outcome=recovery_result["decision"]["outcome"],
                decision_provider=recovery_result["decision"]["provider"],
                rounds=recovery_result["rounds"],
                abstention=recovery_result["abstention"],
            )
            recovered["total_latency_ms"] = recovery_latency
            cases.append(
                {
                    **common,
                    "arms": {
                        "direct": direct,
                        "jev_gated": gated,
                        "jev_gated_recovery": recovered,
                    },
                }
            )
            for name, record in (("direct", direct), ("jev_gated", gated), ("jev_gated_recovery", recovered)):
                arm_records[name].append(record)
    finally:
        if temporary is not None:
            temporary.cleanup()
    return {
        "benchmark": "contextvault-end-to-end-rag-v2",
        "corpus": str(corpus_path),
        "queries": len(cases),
        "k": limit,
        "max_rounds": max_rounds,
        "arms": {name: _summarize(records) for name, records in arm_records.items()},
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the live end-to-end RAG comparison benchmark")
    parser.add_argument("--live", action="store_true", required=True, help="enable network calls")
    parser.add_argument("--corpus", type=Path, default=ROOT / "robust-text-corpus.json")
    parser.add_argument("--queries", type=Path, default=ROOT / "robust-text-queries.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--model", default=os.getenv("OPENROUTER_GENERATION_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.api_key:
        raise SystemExit("--live requires --api-key or OPENROUTER_API_KEY")
    result = run_benchmark(
        args.corpus,
        args.queries,
        generation_provider=OpenRouterGenerationProvider(args.api_key, model=args.model),
        decision_provider=JevDecisionProvider(args.api_key),
        limit=args.limit,
        max_rounds=args.max_rounds,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
