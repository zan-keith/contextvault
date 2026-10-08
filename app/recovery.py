from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any, Protocol

from app.answers import (
    GeneratedAnswer,
    GeneratedClaim,
    GenerationProvider,
    GenerationProviderError,
    build_evidence_manifest,
    validate_generated_answer,
)
from app.db import Database
from app.decisions import DecisionProvider, EvidenceDecision

DEFAULT_MAX_ROUNDS = 2
DEFAULT_RECOVERY_LIMIT = 10

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
        "from", "how", "i", "in", "is", "it", "many", "much", "must", "of", "or", "should",
        "the", "to", "what", "when", "where", "which", "who", "whom", "why", "will", "with",
    }
)


class RecoverySearchProvider(Protocol):
    """Pluggable evidence-recovery search used after a gating rejection."""

    def search(
        self,
        question: str,
        *,
        product: str | None = None,
        version: str | None = None,
        document_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        ...


class DatabaseRecoverySearch:
    """Lexical recovery search over the same metadata-filtered database."""

    def __init__(self, database: Database):
        self.database = database

    def search(
        self,
        question: str,
        *,
        product: str | None = None,
        version: str | None = None,
        document_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self.database.search(
            question,
            product=product,
            version=version,
            document_type=document_type,
            limit=limit,
        )


def _terms(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in _STOPWORDS]


def build_recovery_queries(question: str, *, max_queries: int = 3) -> list[str]:
    """Derive focused follow-up queries from the terms the gate found missing.

    The recovery planner is deliberately lexical: it drops the least specific
    phrasing of the original question and re-asks with the salient terms, so a
    rejection for one missing fact becomes a targeted search instead of an
    abstention.
    """
    terms = _terms(question)
    if not terms:
        return []
    queries: list[str] = [" ".join(terms)]
    if len(terms) > 3:
        queries.append(" ".join(terms[:3]))
        queries.append(" ".join(terms[-3:]))
    unique: list[str] = []
    for query in queries:
        if query and query != question.lower() and query not in unique:
            unique.append(query)
    return unique[:max_queries]


def _merge_candidates(
    initial: Sequence[dict[str, Any]],
    recovered: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Merge recovery hits into the candidate set without path leakage."""
    seen_chunk_ids = {candidate.get("chunk_id") for candidate in initial}
    new_candidates = []
    for candidate in recovered:
        if candidate.get("chunk_id") in seen_chunk_ids:
            continue
        new_candidates.append(candidate)
        seen_chunk_ids.add(candidate.get("chunk_id"))
    # A recovery round exists to surface new evidence; it therefore receives
    # priority while retaining as much initial evidence as the bounded manifest
    # permits.
    return [*new_candidates, *initial][:limit]


def _gate_allows_generation(decision: EvidenceDecision) -> bool:
    if decision.outcome != "answerable":
        return False
    if not decision.trusted or decision.degraded:
        return False
    conflict = decision.conflict_probability
    return conflict is None or conflict < 0.7


async def answer_with_recovery(
    question: str,
    initial_candidates: Sequence[dict[str, Any]],
    *,
    decision_provider: DecisionProvider,
    generation_provider: GenerationProvider,
    recovery_search: RecoverySearchProvider | None = None,
    product: str | None = None,
    version: str | None = None,
    document_type: str | None = None,
    limit: int = 5,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    recovery_limit: int = DEFAULT_RECOVERY_LIMIT,
) -> dict[str, Any]:
    """Answer with a Jev gate plus targeted recovery search after rejection.

    Round 0 evaluates the initial candidates. When the trusted decision is not
    answerable and recovery search is configured, each recovery query adds new
    candidates and the gate is re-evaluated. Generation only ever sees the
    server-built evidence manifest, and every gate outcome is recorded so the
    caller can audit why an answer was or was not produced.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    manifest_candidates: list[dict[str, Any]] = list(initial_candidates)[:limit]
    rounds: list[dict[str, Any]] = []
    decision: EvidenceDecision | None = None
    decision_latency = 0.0
    started = time.perf_counter()

    for round_index in range(max_rounds):
        round_started = time.perf_counter()
        decision_error: str | None = None
        try:
            decision = await decision_provider.evaluate(question, manifest_candidates)
        except Exception as exc:  # noqa: BLE001 - recovery must fail closed, not crash
            decision = None
            decision_error = type(exc).__name__
        decision_latency = round((time.perf_counter() - round_started) * 1000, 2)
        rounds.append(
            {
                "round": round_index,
                "candidate_count": len(manifest_candidates),
                "candidate_names": [candidate.get("name") for candidate in manifest_candidates],
                "decision_outcome": decision.outcome if decision else None,
                "decision_conflict_probability": decision.conflict_probability if decision else None,
                "decision_trusted": decision.trusted if decision else False,
                "decision_degraded": decision.degraded if decision else False,
                "decision_error": decision_error,
                "decision_latency_ms": decision_latency,
                "recovered_new_candidates": 0,
            }
        )
        if decision is not None and _gate_allows_generation(decision):
            break
        if round_index == max_rounds - 1 or recovery_search is None:
            break
        recovered_new = 0
        for recovery_query in build_recovery_queries(question):
            recovered = recovery_search.search(
                recovery_query,
                product=product,
                version=version,
                document_type=document_type,
                limit=recovery_limit,
            )
            before_ids = {candidate.get("chunk_id") for candidate in manifest_candidates}
            manifest_candidates = _merge_candidates(manifest_candidates, recovered, limit=limit)
            recovered_new += sum(
                candidate.get("chunk_id") not in before_ids for candidate in manifest_candidates
            )
            if len(manifest_candidates) >= limit:
                break
        rounds[-1]["recovered_new_candidates"] = recovered_new
        if recovered_new == 0:
            break

    total_latency = round((time.perf_counter() - started) * 1000, 2)
    result: dict[str, Any] = {
        "query": question,
        "rounds": rounds,
        "recovery_enabled": recovery_search is not None,
        "candidates": [
            {
                "evidence_id": item["evidence_id"],
                "name": item.get("name"),
                "source_hash": item.get("source_hash"),
            }
            for item in build_evidence_manifest(manifest_candidates)
        ],
        "decision": {
            "outcome": decision.outcome if decision else None,
            "confidence": decision.confidence if decision else None,
            "conflict_probability": decision.conflict_probability if decision else None,
            "trusted": decision.trusted if decision else False,
            "degraded": decision.degraded if decision else False,
            "provider": decision.provider if decision else None,
            "provider_version": decision.provider_version if decision else None,
            "reasons": list(decision.reasons) if decision else [],
        },
        "answer": None,
        "abstention": None,
        "generation": None,
        "total_latency_ms": total_latency,
    }
    if decision is None or not _gate_allows_generation(decision) or not manifest_candidates:
        reason = "decision_provider_error"
        if decision is not None:
            if decision.outcome == "answerable":
                reason = "untrusted_or_degraded_decision"
            elif decision.conflict_probability is not None and decision.conflict_probability >= 0.7:
                reason = "conflicting_evidence"
            elif recovery_search is not None:
                reason = "insufficient_evidence_after_recovery"
            else:
                reason = str(decision.outcome)
        result["abstention"] = {
            "reason": reason,
            "action": "Review the evidence before relying on an answer.",
        }
        return result

    manifest = build_evidence_manifest(manifest_candidates)
    try:
        generated = await generation_provider.generate(question, manifest)
        validated = validate_generated_answer(generated, manifest)
    except Exception as exc:  # noqa: BLE001 - generation failures fail closed
        result["abstention"] = {
            "reason": (
                "generation_provider_unavailable"
                if type(exc).__name__ == "GenerationUnavailableError"
                else "generation_provider_error"
                if isinstance(exc, GenerationProviderError)
                else "invalid_generated_answer"
            ),
            "action": "The answer provider failed; review the evidence or try again later.",
        }
        return result
    result["answer"] = {
        "text": validated.answer,
        "claims": [claim.model_dump() for claim in validated.claims],
    }
    result["generation"] = {
        "provider": validated.provider,
        "provider_version": validated.provider_version,
        "latency_ms": validated.latency_ms,
        "input_tokens": validated.input_tokens,
        "output_tokens": validated.output_tokens,
        "cost_usd": validated.cost_usd,
    }
    return result


__all__ = [
    "DatabaseRecoverySearch",
    "GeneratedAnswer",
    "GeneratedClaim",
    "RecoverySearchProvider",
    "answer_with_recovery",
    "build_recovery_queries",
]
