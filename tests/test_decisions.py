import asyncio

from app.decisions import RuleDecisionProvider


def test_rules_provider_marks_matching_evidence_answerable():
    provider = RuleDecisionProvider()

    result = asyncio.run(
        provider.evaluate(
            "How do I investigate a vacuum alarm?",
            [
                {
                    "file_id": 1,
                    "name": "maintenance.txt",
                    "description": "Vacuum alarm troubleshooting",
                    "text": "Inspect the vacuum sensor and check the seal.",
                }
            ],
        )
    )

    assert result.provider == "rules"
    assert result.outcome == "answerable"
    assert result.confidence > 0.5


def test_rules_provider_abstains_without_candidates():
    result = asyncio.run(RuleDecisionProvider().evaluate("unknown question", []))

    assert result.provider == "rules"
    assert result.outcome == "insufficient_evidence"
    assert result.confidence == 0.0
