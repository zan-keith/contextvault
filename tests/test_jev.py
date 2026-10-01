import asyncio
import json

from app.decisions import (
    DecisionResult,
    FallbackDecisionProvider,
    JevDecisionProvider,
    RuleDecisionProvider,
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
    assert provider.request_body["model"] == "jev-latest"
    assert provider.request_body["questions"]["answerable"]["type"] == "noul"


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
