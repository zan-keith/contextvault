import asyncio
import json

from app.decisions import (
    FallbackDecisionProvider,
    JevDecisionProvider,
    RuleDecisionProvider,
    provider_from_environment,
)


def test_jev_provider_builds_typed_decision_request_without_network():
    class StubJev(JevDecisionProvider):
        def _request(self, body: bytes) -> bytes:
            self.request_body = json.loads(body)
            return json.dumps(
                {
                    "answers": {
                        "answerable": {"type": "noul", "noul": 0.91},
                        "conflict": {"type": "noul", "noul": 0.04},
                    }
                }
            ).encode()

    provider = StubJev("test-key")
    result = asyncio.run(provider.evaluate("How do I calibrate X200?", [{"file_id": 1, "text": "Calibrate X200."}]))

    assert result.provider == "jev"
    assert result.outcome == "answerable"
    assert result.confidence == 0.91
    assert provider.request_body["model"] == "typesafe/jev-1.13"
    assert provider.request_body["questions"]["answerable"]["type"] == "noul"
    assert "complete set" in provider.request_body["questions"]["answerable"]["instructions"]


def test_jev_provider_exposes_probabilities_and_honours_configured_thresholds():
    class StubJev(JevDecisionProvider):
        def _request(self, body: bytes) -> bytes:
            return json.dumps(
                {
                    "answers": {
                        "answerable": {"type": "noul", "noul": 0.81},
                        "conflict": {"type": "noul", "noul": 0.75},
                    }
                }
            ).encode()

    provider = StubJev("test-key", conflict_threshold=0.8)
    result = asyncio.run(provider.evaluate("question", [{"text": "evidence"}]))

    assert result.outcome == "answerable"
    assert result.confidence == 0.81
    assert result.conflict_probability == 0.75
    assert result.provider_version == "typesafe/jev-1.13"
    assert result.latency_ms is not None
    assert result.latency_ms >= 0


def test_jev_provider_falls_back_to_rules_on_provider_failure():
    class BrokenProvider:
        async def evaluate(self, question, candidates):
            raise RuntimeError("offline")

    result = asyncio.run(
        FallbackDecisionProvider(BrokenProvider(), RuleDecisionProvider()).evaluate(
            "vacuum alarm",
            [{"name": "manual", "text": "Inspect the vacuum sensor."}],
        )
    )

    assert result.provider == "rules-fallback"
    assert result.outcome == "answerable"
    assert "RuntimeError" in result.reasons[0]


def test_provider_uses_openrouter_key_and_keeps_rules_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    provider = provider_from_environment()

    assert isinstance(provider, FallbackDecisionProvider)
    assert isinstance(provider.primary, JevDecisionProvider)
    assert isinstance(provider.fallback, RuleDecisionProvider)
