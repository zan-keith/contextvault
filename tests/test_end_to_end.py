from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.answers import GeneratedAnswer, GeneratedClaim
from app.decisions import DecisionResult
from app.main import create_app
from app.recovery import (
    DatabaseRecoverySearch,
    answer_with_recovery,
    build_recovery_queries,
)
from benchmarks.end_to_end import run_benchmark


class ScriptedDecision:
    """Returns scripted outcomes per call so recovery rounds are deterministic."""

    def __init__(self, outcomes: list[DecisionResult]):
        self.outcomes = list(outcomes)
        self.calls: list[list[dict[str, Any]]] = []

    async def evaluate(self, _question: str, candidates: list[dict[str, Any]]) -> DecisionResult:
        self.calls.append(candidates)
        index = min(len(self.calls) - 1, len(self.outcomes) - 1)
        return self.outcomes[index]


def _trusted(outcome: str, conflict: float | None = None) -> DecisionResult:
    return DecisionResult(
        outcome=outcome,
        confidence=0.9,
        reasons=["test"],
        provider="jev",
        trusted=True,
        conflict_probability=conflict,
    )


class FakeGeneration:
    def __init__(self):
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def generate(self, question, evidence):
        self.calls.append((question, evidence))
        return GeneratedAnswer(
            answer="Follow the documented procedure.",
            claims=(GeneratedClaim(text="Follow the documented procedure.", citation_ids=(evidence[0]["evidence_id"],)),),
            input_tokens=10,
            output_tokens=4,
            cost_usd=0.01,
        )


def _candidate(name: str, chunk_id: int, text: str = "Some procedure text.") -> dict[str, Any]:
    return {
        "file_id": chunk_id,
        "chunk_id": chunk_id,
        "name": name,
        "description": name,
        "text": text,
        "product": None,
        "version": None,
        "document_type": "runbook",
        "status": "active",
        "sha256": f"hash-{chunk_id}",
        "ordinal": 0,
    }


class StaticRecoverySearch:
    def __init__(self, results: list[dict[str, Any]]):
        self.results = results
        self.queries: list[str] = []

    def search(self, question, *, product=None, version=None, document_type=None, limit=10):
        self.queries.append(question)
        return list(self.results)


def test_build_recovery_queries_drops_stopwords_and_duplicates():
    queries = build_recovery_queries("How many times should the system retry a failed card capture before abandoning it?")
    assert queries
    assert all("the" not in query.split() for query in queries)
    assert len(queries) == len(set(queries))
    assert build_recovery_queries("???") == []


def test_recovery_search_recovers_missing_evidence_and_generates(tmp_path):
    from app.db import Database

    db = Database(tmp_path / "recovery.db")
    db.create_files(
        [
            {
                "name": "outage.txt",
                "description": "Card capture outage runbook",
                "content": "When a card capture cannot reach the payment network, retain the authorization and enqueue the capture for retry.",
                "product": "finance",
                "version": "v2",
                "document_type": "runbook",
                "status": "active",
            },
            {
                "name": "retry-window.txt",
                "description": "Retry schedule for queued card captures",
                "content": "Queued card captures are retried every five minutes for up to two hours before escalation.",
                "product": "finance",
                "version": "v2",
                "document_type": "runbook",
                "status": "active",
            },
        ]
    )
    initial = db.search("How many times should the system retry a failed card capture before abandoning it?", limit=1)
    assert all(item["name"] == "outage.txt" for item in initial)

    decision = ScriptedDecision([_trusted("insufficient_evidence"), _trusted("answerable")])
    generation = FakeGeneration()
    result = asyncio.run(
        answer_with_recovery(
            "How many times should the system retry a failed card capture before abandoning it?",
            initial,
            decision_provider=decision,
            generation_provider=generation,
            recovery_search=DatabaseRecoverySearch(db),
            limit=5,
            max_rounds=2,
        )
    )

    assert result["answer"] is not None
    assert len(result["rounds"]) == 2
    assert result["rounds"][0]["recovered_new_candidates"] >= 1
    assert result["recovery_enabled"] is True
    assert result["decision"]["outcome"] == "answerable"
    assert len(generation.calls) == 1
    names = {item["name"] for item in generation.calls[0][1]}
    assert "retry-window.txt" in names
    assert all("source_uri" not in item for item in generation.calls[0][1])


def test_recovery_can_replace_a_full_initial_candidate_set_with_new_evidence():
    initial = [_candidate(f"old-{index}.txt", index) for index in range(1, 6)]
    fresh = _candidate("fresh.txt", 6)
    decision = ScriptedDecision([_trusted("insufficient_evidence"), _trusted("answerable")])
    generation = FakeGeneration()
    search = StaticRecoverySearch([fresh])

    result = asyncio.run(
        answer_with_recovery(
            "What detailed missing procedure should we follow today?",
            initial,
            decision_provider=decision,
            generation_provider=generation,
            recovery_search=search,
            limit=5,
            max_rounds=2,
        )
    )

    assert result["answer"] is not None
    assert result["rounds"][0]["recovered_new_candidates"] == 1
    assert "fresh.txt" in {item["name"] for item in generation.calls[0][1]}


def test_recovery_abstains_when_nothing_new_is_found():
    decision = ScriptedDecision([_trusted("insufficient_evidence")])
    generation = FakeGeneration()
    search = StaticRecoverySearch([])
    result = asyncio.run(
        answer_with_recovery(
            "What is the maximum capture amount?",
            [_candidate("outage.txt", 1)],
            decision_provider=decision,
            generation_provider=generation,
            recovery_search=search,
            limit=5,
            max_rounds=3,
        )
    )
    assert result["answer"] is None
    assert result["abstention"]["reason"] == "insufficient_evidence_after_recovery"
    assert generation.calls == []
    assert len(result["rounds"]) == 1


def test_recovery_does_not_generate_on_conflict_even_after_recovery():
    decision = ScriptedDecision([_trusted("review", conflict=0.9), _trusted("review", conflict=0.9)])
    generation = FakeGeneration()
    search = StaticRecoverySearch([_candidate("more.txt", 2)])
    result = asyncio.run(
        answer_with_recovery(
            "What should we do when the router stops forwarding traffic?",
            [_candidate("restart.txt", 1)],
            decision_provider=decision,
            generation_provider=generation,
            recovery_search=search,
            limit=5,
            max_rounds=2,
        )
    )
    assert result["answer"] is None
    assert result["abstention"]["reason"] == "conflicting_evidence"
    assert generation.calls == []


def test_recovery_fails_closed_on_decision_error():
    class BrokenDecision:
        async def evaluate(self, _question, _candidates):
            raise TimeoutError("jev timed out")

    generation = FakeGeneration()
    search = StaticRecoverySearch([_candidate("more.txt", 2)])
    result = asyncio.run(
        answer_with_recovery(
            "What should we do?",
            [_candidate("doc.txt", 1)],
            decision_provider=BrokenDecision(),
            generation_provider=generation,
            recovery_search=search,
            limit=5,
            max_rounds=2,
        )
    )
    assert result["answer"] is None
    assert result["abstention"]["reason"] == "decision_provider_error"
    assert generation.calls == []


def test_recovery_rejects_invalid_citation_after_recovery(tmp_path):
    from app.db import Database

    db = Database(tmp_path / "invalid.db")
    db.create_files(
        [
            {
                "name": "outage.txt",
                "description": "Card capture outage runbook",
                "content": "Retain the authorization and enqueue the capture for retry.",
                "product": "finance",
                "version": "v2",
                "document_type": "runbook",
                "status": "active",
            }
        ]
    )
    initial = db.search("retry a failed card capture", limit=1)
    decision = ScriptedDecision([_trusted("answerable")])

    class BadGenerator:
        async def generate(self, _question, _evidence):
            return GeneratedAnswer(
                answer="Unsafe.",
                claims=[GeneratedClaim(text="Unsafe.", citation_ids=["ev-does-not-exist"])],
            )

    result = asyncio.run(
        answer_with_recovery(
            "retry a failed card capture",
            initial,
            decision_provider=decision,
            generation_provider=BadGenerator(),
            recovery_search=DatabaseRecoverySearch(db),
            limit=5,
        )
    )
    assert result["answer"] is None
    assert result["abstention"]["reason"] == "invalid_generated_answer"


def test_recovery_api_endpoint_returns_answer_or_abstention(tmp_path):
    class AnswerableDecision:
        async def evaluate(self, _question, _candidates):
            return _trusted("answerable")

    generation = FakeGeneration()
    client = TestClient(
        create_app(
            tmp_path / "api.db",
            decision_provider=AnswerableDecision(),
            generation_provider=generation,
        )
    )
    client.post(
        "/files",
        json={
            "name": "runbook.txt",
            "description": "Vacuum alarm runbook",
            "content": "Inspect the vacuum sensor and check the seal.",
            "product": "X200",
            "version": "B",
            "document_type": "maintenance",
            "status": "active",
        },
    )
    response = client.post(
        "/answers/recovery",
        json={"question": "How do I investigate a vacuum alarm?", "strategy": "fts"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is not None
    assert body["recovery_enabled"] is True
    assert body["rounds"]
    assert len(generation.calls) == 1


def test_recovery_api_abstains_closed_without_candidates(tmp_path):
    class AnswerableDecision:
        async def evaluate(self, _question, _candidates):
            return _trusted("answerable")

    generation = FakeGeneration()
    client = TestClient(
        create_app(
            tmp_path / "empty.db",
            decision_provider=AnswerableDecision(),
            generation_provider=generation,
        )
    )
    response = client.post("/answers/recovery", json={"question": "Unknown topic?", "strategy": "fts"})
    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["abstention"]["reason"]
    assert generation.calls == []


class FakeJev:
    provider_version = "jev-test"

    async def evaluate(self, _question, candidates):
        return DecisionResult(
            outcome="answerable" if candidates else "insufficient_evidence",
            confidence=0.99,
            reasons=["test"],
            provider="jev",
            provider_version=self.provider_version,
            trusted=True,
            input_tokens=3,
            cost_usd=0.02,
        )


def test_arms_share_fts_candidates_and_recovery_arm_has_three_lanes(tmp_path):
    generation = FakeGeneration()
    result = run_benchmark(
        Path("benchmarks/robust-text-corpus.json"),
        Path("benchmarks/robust-text-queries.json"),
        generation_provider=generation,
        decision_provider=FakeJev(),
        database_path=tmp_path / "benchmark.db",
        limit=5,
    )

    assert result["queries"] == 30
    assert set(result["arms"]) == {"direct", "jev_gated", "jev_gated_recovery"}
    assert all(case["retrieved_files"] == case["arms"]["direct"]["retrieved_files"] for case in result["cases"])
    assert all(case["arms"]["direct"]["action"] == "answer" for case in result["cases"] if case["retrieved_files"])
    assert result["arms"]["direct"]["total_tokens"] == 28 * 14
    assert result["arms"]["jev_gated"]["total_tokens"] == 28 * 14 + 30 * 3
    assert result["arms"]["jev_gated_recovery"]["answers"] >= result["arms"]["jev_gated"]["answers"]
    expected_generation_calls = (
        result["arms"]["direct"]["answers"]
        + result["arms"]["jev_gated"]["answers"]
        + result["arms"]["jev_gated_recovery"]["answers"]
    )
    assert len(generation.calls) == expected_generation_calls


def test_recovery_arm_recovers_multi_hop_case(tmp_path):
    generation = FakeGeneration()
    result = run_benchmark(
        Path("benchmarks/robust-text-corpus.json"),
        Path("benchmarks/robust-text-queries.json"),
        generation_provider=generation,
        decision_provider=FakeJev(),
        database_path=tmp_path / "benchmark.db",
        limit=5,
    )
    multi_hop = next(case for case in result["cases"] if case["case_id"] == "capture-retry-count")
    recovered_names = {
        name
        for round_item in multi_hop["arms"]["jev_gated_recovery"]["rounds"]
        for name in round_item["candidate_names"]
    }
    assert "finance-capture-retry-window.txt" in recovered_names
    assert multi_hop["arms"]["jev_gated_recovery"]["action"] == "answer"


def test_jev_gated_arm_abstains_on_review_and_does_not_generate(tmp_path):
    class ReviewJev(FakeJev):
        async def evaluate(self, question, candidates):
            if "edge router" in question:
                return DecisionResult("review", 1.0, ["conflict"], "jev", trusted=True)
            return await super().evaluate(question, candidates)

    generation = FakeGeneration()
    result = run_benchmark(
        Path("benchmarks/robust-text-corpus.json"),
        Path("benchmarks/robust-text-queries.json"),
        generation_provider=generation,
        decision_provider=ReviewJev(),
        database_path=tmp_path / "benchmark.db",
    )

    edge = next(case for case in result["cases"] if case["case_id"] == "edge-router-conflict")
    assert edge["arms"]["jev_gated"]["action"] == "abstain"
    assert edge["arms"]["jev_gated"]["generation_called"] is False
    assert edge["arms"]["jev_gated_recovery"]["action"] == "abstain"
    assert result["arms"]["jev_gated"]["abstentions"] >= 1


def test_live_cli_requires_explicit_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from benchmarks.end_to_end import build_parser, main

    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        main(["--live"])
