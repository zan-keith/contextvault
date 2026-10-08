from __future__ import annotations

import asyncio

import pytest

from app.answers import GeneratedAnswer, GeneratedClaim
from app.db import Database
from app.decisions import DecisionResult
from app.recovery import (
    DatabaseRecoverySearch,
    answer_with_recovery,
)


def _doc(name: str, content: str, **metadata) -> dict:
    return {
        "name": name,
        "description": name,
        "content": content,
        "product": metadata.get("product"),
        "version": metadata.get("version"),
        "document_type": metadata.get("document_type", "runbook"),
        "status": metadata.get("status", "active"),
    }


class RejectThenApprove:
    """Approves only when every required file is present; otherwise rejects."""

    def __init__(self, required_names: set[str]):
        self.required = required_names
        self.calls = 0

    async def evaluate(self, _question, candidates):
        self.calls += 1
        names = {candidate["name"] for candidate in candidates}
        if self.required and self.required <= names:
            return DecisionResult("answerable", 0.95, ["recovered"], "jev", trusted=True)
        return DecisionResult("insufficient_evidence", 0.3, ["missing"], "jev", trusted=True)


class EchoGeneration:
    def __init__(self):
        self.calls = []

    async def generate(self, _question, evidence):
        self.calls.append(evidence)
        return GeneratedAnswer(
            answer=evidence[0]["text"],
            claims=[GeneratedClaim(text=evidence[0]["text"][:80], citation_ids=[evidence[0]["evidence_id"]])],
        )


@pytest.fixture()
def two_file_db(tmp_path):
    db = Database(tmp_path / "recovery.db")
    db.create_files(
        [
            _doc(
                "outage.txt",
                "When a card capture cannot reach the payment network, retain the authorization and enqueue the capture for retry.",
                product="finance",
                version="v2",
            ),
            _doc(
                "retry-window.txt",
                "Queued card captures are retried every five minutes for up to two hours before escalation to the on-call engineer.",
                product="finance",
                version="v2",
            ),
        ]
    )
    return db


def test_recovery_finds_second_hop_fact(two_file_db):
    question = (
        "How many times should the system retry a failed card capture before abandoning it?"
    )
    initial = two_file_db.search(question, product="finance", version="v2", limit=1)
    assert [row["name"] for row in initial] == ["outage.txt"]

    decision = RejectThenApprove({"outage.txt", "retry-window.txt"})
    generation = EchoGeneration()
    result = asyncio.run(
        answer_with_recovery(
            question,
            initial,
            decision_provider=decision,
            generation_provider=generation,
            recovery_search=DatabaseRecoverySearch(two_file_db),
            product="finance",
            version="v2",
            limit=5,
            max_rounds=2,
        )
    )

    assert result["answer"] is not None
    assert decision.calls == 2
    assert len(result["rounds"]) == 2
    assert result["rounds"][0]["recovered_new_candidates"] == 1
    generator_names = {item["name"] for item in generation.calls[0]}
    assert "retry-window.txt" in generator_names


def test_recovery_does_not_rescue_unanswerable_question(two_file_db):
    question = "What is the maximum amount that can be captured in a single card transaction?"
    initial = two_file_db.search(question, product="finance", version="v2", limit=5)
    decision = RejectThenApprove(set())  # never approves
    generation = EchoGeneration()
    result = asyncio.run(
        answer_with_recovery(
            question,
            initial,
            decision_provider=decision,
            generation_provider=generation,
            recovery_search=DatabaseRecoverySearch(two_file_db),
            product="finance",
            version="v2",
            limit=5,
            max_rounds=2,
        )
    )
    assert result["answer"] is None
    assert result["abstention"]["reason"] == "insufficient_evidence_after_recovery"
    assert generation.calls == []


def test_recovery_respects_metadata_filters(two_file_db):
    question = "How many times should the system retry a failed card capture before abandoning it?"
    initial = two_file_db.search(question, product="web", version="v9", limit=5)
    assert initial == []
    decision = RejectThenApprove(set())
    generation = EchoGeneration()
    result = asyncio.run(
        answer_with_recovery(
            question,
            initial,
            decision_provider=decision,
            generation_provider=generation,
            recovery_search=DatabaseRecoverySearch(two_file_db),
            product="web",
            version="v9",
            limit=5,
            max_rounds=2,
        )
    )
    assert result["answer"] is None
    assert generation.calls == []


def test_recovery_audit_reports_all_expected_files_reachable():
    from benchmarks.recovery_audit import main as audit_main

    audit_main()  # must not raise


def test_offline_diagnostic_three_arms(tmp_path):
    from benchmarks.offline_diagnostic import run_offline_diagnostic

    result = run_offline_diagnostic()
    assert result["queries"] == 30
    arms = result["arms"]
    assert set(arms) == {"direct", "jev_gated", "jev_gated_recovery"}
    # The gated arm must never answer an insufficient/review case.
    assert arms["jev_gated"]["action_correctness"] == 1.0
    # Recovery must match the gated arm's coverage without lowering correctness.
    assert arms["jev_gated_recovery"]["action_correctness"] == 1.0
    assert arms["jev_gated_recovery"]["answers"] >= arms["jev_gated"]["answers"]
    # Direct RAG answers unsupported cases; the gate prevents that.
    assert arms["direct"]["action_correctness"] < arms["jev_gated"]["action_correctness"]
