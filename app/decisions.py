from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, Protocol

DecisionOutcome = Literal[
    "answerable",
    "partial",
    "insufficient_evidence",
    "review",
    "conflict",
    "stale",
    "scope_mismatch",
]


@dataclass(frozen=True)
class RequirementDecision:
    requirement_id: str
    text: str
    status: str
    support_probability: float | None = None


@dataclass(frozen=True)
class EvidenceDecision:
    outcome: DecisionOutcome | str
    confidence: float
    reasons: tuple[str, ...] | list[str]
    provider: str
    conflict_probability: float | None = None
    coverage: float | None = None
    applicability: float | None = None
    freshness: float | None = None
    provider_version: str = "unknown"
    latency_ms: float | None = None
    input_tokens: int | None = None
    cost_usd: float | None = None
    requirements: tuple[RequirementDecision, ...] | list[RequirementDecision] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "requirements", tuple(self.requirements))

    @property
    def support_probability(self) -> float:
        return self.confidence


# Compatibility name retained while callers migrate to the evidence-oriented model.
DecisionResult = EvidenceDecision


class DecisionProvider(Protocol):
    async def evaluate(self, question: str, candidates: list[dict[str, Any]]) -> DecisionResult:
        ...


class RuleDecisionProvider:
    """Transparent local fallback used for tests and development."""

    provider_name = "rules"
    provider_version = "rules-v1"
    _stopwords: ClassVar[set[str]] = {"a", "an", "and", "are", "do", "for", "how", "i", "is", "the", "to", "what"}

    @classmethod
    def _terms(cls, value: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9]+", value.lower())
            if term not in cls._stopwords
        }

    async def evaluate(self, question: str, candidates: list[dict[str, Any]]) -> DecisionResult:
        if not candidates:
            return DecisionResult(
                outcome="insufficient_evidence",
                confidence=0.0,
                reasons=["No active evidence matched the query."],
                provider=self.provider_name,
                provider_version=self.provider_version,
            )

        query_terms = self._terms(question)
        best_overlap = 0.0
        best_name = candidates[0].get("name", "candidate")
        for candidate in candidates:
            evidence = " ".join(
                str(candidate.get(field, ""))
                for field in ("name", "description", "text", "product", "version")
            )
            evidence_terms = self._terms(evidence)
            overlap = len(query_terms & evidence_terms) / max(len(query_terms), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = candidate.get("name", "candidate")

        confidence = round(min(0.99, 0.25 + best_overlap * 0.75), 3)
        if best_overlap >= 0.25:
            return DecisionResult(
                outcome="answerable",
                confidence=confidence,
                reasons=[f"Lexical evidence matched {best_name}."],
                provider=self.provider_name,
                provider_version=self.provider_version,
            )
        return DecisionResult(
            outcome="insufficient_evidence",
            confidence=confidence,
            reasons=["Retrieved evidence has insufficient query-term coverage."],
            provider=self.provider_name,
            provider_version=self.provider_version,
        )


class JevDecisionProvider:
    """OpenRouter adapter for TypeSafe Jev's typed Decisions API.

    Jev verifies evidence sufficiency and conflicts; it deliberately does not
    generate an answer. OpenRouter's `typesafe/jev-router` is reserved for a
    separate future chat-model routing step.
    """

    provider_name = "jev"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/alpha/decisions",
        model: str = "typesafe/jev-1.13",
        timeout: float = 10.0,
        answerability_threshold: float = 0.7,
        conflict_threshold: float = 0.7,
    ):
        if not api_key.strip():
            raise ValueError("Jev API key must not be blank")
        if not 0 <= answerability_threshold <= 1:
            raise ValueError("answerability_threshold must be between 0 and 1")
        if not 0 <= conflict_threshold <= 1:
            raise ValueError("conflict_threshold must be between 0 and 1")
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.answerability_threshold = answerability_threshold
        self.conflict_threshold = conflict_threshold

    def _request(self, body: bytes) -> bytes:
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    async def evaluate(self, question: str, candidates: list[dict[str, Any]]) -> DecisionResult:
        state = {
            "question": question,
            "candidates": [
                {
                    "file_id": candidate.get("file_id"),
                    "name": candidate.get("name"),
                    "description": candidate.get("description"),
                    "product": candidate.get("product"),
                    "version": candidate.get("version"),
                    "document_type": candidate.get("document_type"),
                    "text": candidate.get("text"),
                }
                for candidate in candidates
            ],
        }
        body = json.dumps(
            {
                "state": state,
                "model": self.model,
                "questions": {
                    "answerable": {
                        "type": "noul",
                        "instructions": "Does at least one candidate passage directly support an answer to the question?",
                        "criteria": {
                            "true": "The candidates contain direct, applicable evidence.",
                            "false": "The candidates are missing, irrelevant, or insufficient.",
                        },
                    },
                    "conflict": {
                        "type": "noul",
                        "instructions": "Do the candidate passages materially conflict with each other?",
                        "criteria": {
                            "true": "The passages give incompatible technical guidance.",
                            "false": "The passages are consistent or non-overlapping.",
                        },
                    },
                },
            }
        ).encode()

        started_at = time.perf_counter()
        try:
            raw = await asyncio.to_thread(self._request, body)
            response = json.loads(raw)
            answers = response.get("answers", {})
            answerability = float(answers.get("answerable", {}).get("noul", 0.0))
            conflict = float(answers.get("conflict", {}).get("noul", 0.0))
        except (OSError, urllib.error.URLError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("Jev decision request failed") from exc

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
        raw_input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        raw_cost = usage.get("cost", response.get("cost"))
        try:
            input_tokens = int(raw_input_tokens) if raw_input_tokens is not None else None
        except (TypeError, ValueError):
            input_tokens = None
        try:
            cost_usd = float(raw_cost) if raw_cost is not None else None
        except (TypeError, ValueError):
            cost_usd = None

        if conflict >= self.conflict_threshold:
            outcome = "review"
            reasons = ["Jev detected potentially conflicting evidence."]
        elif answerability >= self.answerability_threshold:
            outcome = "answerable"
            reasons = ["Jev judged the retrieved evidence sufficient."]
        else:
            outcome = "insufficient_evidence"
            reasons = ["Jev judged the retrieved evidence insufficient."]
        return DecisionResult(
            outcome=outcome,
            confidence=round(answerability, 3),
            reasons=reasons,
            provider=self.provider_name,
            provider_version=self.model,
            conflict_probability=round(conflict, 3),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            cost_usd=cost_usd,
        )


class FallbackDecisionProvider:
    def __init__(self, primary: DecisionProvider, fallback: DecisionProvider):
        self.primary = primary
        self.fallback = fallback

    async def evaluate(self, question: str, candidates: list[dict[str, Any]]) -> DecisionResult:
        try:
            return await self.primary.evaluate(question, candidates)
        except Exception as exc:  # noqa: BLE001 - primary adapters must never break query retrieval
            fallback_result = await self.fallback.evaluate(question, candidates)
            return replace(
                fallback_result,
                provider=f"{fallback_result.provider}-fallback",
                reasons=[f"Primary decision provider unavailable: {type(exc).__name__}.", *fallback_result.reasons],
            )


def provider_from_environment() -> DecisionProvider:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return RuleDecisionProvider()
    return FallbackDecisionProvider(JevDecisionProvider(api_key), RuleDecisionProvider())
