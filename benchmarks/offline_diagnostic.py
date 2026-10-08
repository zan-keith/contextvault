from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.answers import GeneratedAnswer, GeneratedClaim, build_evidence_manifest
from app.db import Database
from app.decisions import DecisionResult
from app.recovery import DatabaseRecoverySearch, answer_with_recovery, build_recovery_queries

CORPUS = Path("benchmarks/robust-text-corpus.json")
QUERIES = Path("benchmarks/robust-text-queries.json")


class CoverageJev:
    """Offline stand-in for Jev: approves only when every expected file is present."""

    provider_version = "coverage-jev-offline"

    def __init__(self, expected_by_case: dict[str, set[str]]):
        self.expected_by_case = expected_by_case
        self.current_question: str | None = None

    async def evaluate(self, question: str, candidates: list[dict[str, Any]]) -> DecisionResult:
        names = {candidate["name"] for candidate in candidates}
        expected = self.expected_by_case.get(question, set())
        covered = expected <= names
        combined_text = " ".join(candidate["text"] for candidate in candidates)
        conflict = "never restart" in combined_text and "restart the router once" in combined_text
        if conflict:
            return DecisionResult("review", 0.2, ["conflicting guidance"], "jev", trusted=True, conflict_probability=0.95)
        if covered and expected:
            return DecisionResult("answerable", 0.95, ["all expected evidence present"], "jev", trusted=True)
        return DecisionResult("insufficient_evidence", 0.3, ["missing expected evidence"], "jev", trusted=True)


class ExtractiveGeneration:
    """Deterministic generator that quotes the first candidate and cites it."""

    async def generate(self, question: str, evidence: list[dict[str, Any]]) -> GeneratedAnswer:
        first = evidence[0]
        claim_text = first["text"][:120]
        return GeneratedAnswer(
            answer=first["text"],
            claims=[GeneratedClaim(text=claim_text, citation_ids=[first["evidence_id"]])],
            provider="offline-extractive",
            provider_version="v1",
            latency_ms=1.0,
            input_tokens=len(question.split()) + sum(len(item["text"].split()) for item in evidence),
            output_tokens=len(first["text"].split()),
            cost_usd=0.0,
        )


def run_offline_diagnostic() -> dict[str, Any]:
    corpus = json.loads(CORPUS.read_text())
    queries = json.loads(QUERIES.read_text())
    import tempfile

    with tempfile.TemporaryDirectory(prefix="contextvault-offline-e2e-") as directory:
        db = Database(Path(directory) / "offline.db")
        db.create_files(corpus)
        expected_by_case = {query["question"]: set(query.get("expected_files", [])) for query in queries}
        jev = CoverageJev(expected_by_case)
        generation = ExtractiveGeneration()
        recovery_search = DatabaseRecoverySearch(db)

        cases = []
        for query in queries:
            question = query["question"]
            expected_files = set(query.get("expected_files", []))
            expected_outcome = query.get("expected_outcome") or ("answerable" if expected_files else "insufficient_evidence")
            initial = db.search(
                question,
                product=query.get("product"),
                version=query.get("version"),
                document_type=query.get("document_type"),
                limit=5,
            )

            direct = asyncio.run(generation.generate(question, build_evidence_manifest(initial))) if initial else None
            direct_action = "answer" if direct is not None else "abstain"
            gated_decision = asyncio.run(jev.evaluate(question, initial))
            gated_allowed = (
                gated_decision.outcome == "answerable"
                and gated_decision.trusted
                and not gated_decision.degraded
                and (gated_decision.conflict_probability is None or gated_decision.conflict_probability < 0.7)
            )
            gated = asyncio.run(generation.generate(question, build_evidence_manifest(initial))) if gated_allowed else None
            gated_action = "answer" if gated is not None else "abstain"

            recovery = asyncio.run(
                answer_with_recovery(
                    question,
                    initial,
                    decision_provider=jev,
                    generation_provider=generation,
                    recovery_search=recovery_search,
                    product=query.get("product"),
                    version=query.get("version"),
                    document_type=query.get("document_type"),
                    limit=5,
                    max_rounds=2,
                )
            )
            recovery_action = "answer" if recovery["answer"] is not None else "abstain"

            cases.append(
                {
                    "case_id": query.get("id"),
                    "question": question,
                    "expected_outcome": expected_outcome,
                    "expected_files": sorted(expected_files),
                    "initial_files": [candidate["name"] for candidate in initial],
                    "direct": direct_action,
                    "jev_gated": gated_action,
                    "jev_gated_recovery": recovery_action,
                    "recovery_rounds": len(recovery["rounds"]),
                    "recovery_new_candidates": sum(r["recovered_new_candidates"] for r in recovery["rounds"]),
                    "final_candidates": [item["name"] for item in recovery["candidates"]],
                }
            )

    def summarize(action_key: str) -> dict[str, Any]:
        correct = [
            case[action_key] == ("answer" if case["expected_outcome"] == "answerable" else "abstain")
            for case in cases
        ]
        answered = sum(case[action_key] == "answer" for case in cases)
        return {
            "action_correctness": round(sum(correct) / len(correct), 4) if correct else None,
            "answers": answered,
            "abstentions": len(cases) - answered,
        }

    return {
        "benchmark": "contextvault-end-to-end-rag-offline-diagnostic-v2",
        "corpus": str(CORPUS),
        "queries": len(cases),
        "k": 5,
        "max_rounds": 2,
        "arms": {
            "direct": summarize("direct"),
            "jev_gated": summarize("jev_gated"),
            "jev_gated_recovery": summarize("jev_gated_recovery"),
        },
        "recovery_queries_sample": build_recovery_queries(queries[0]["question"]),
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(run_offline_diagnostic(), indent=2))
