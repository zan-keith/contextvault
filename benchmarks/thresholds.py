from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _outcome(
    answerability_probability: float,
    conflict_probability: float,
    *,
    answerability_threshold: float,
    conflict_threshold: float,
) -> str:
    if conflict_probability >= conflict_threshold:
        return "review"
    if answerability_probability >= answerability_threshold:
        return "answerable"
    return "insufficient_evidence"


def summarize_thresholds(
    rows: list[dict[str, Any]],
    *,
    answerability_thresholds: list[float],
    conflict_threshold: float,
) -> list[dict[str, float | int | None]]:
    """Summarise threshold trade-offs from one fixed set of Jev probabilities."""
    valid = [
        row
        for row in rows
        if row.get("decision_error") is None
        and row.get("decision_confidence") is not None
        and row.get("decision_conflict_probability") is not None
    ]
    if not 0 <= conflict_threshold <= 1:
        raise ValueError("conflict_threshold must be between 0 and 1")

    summaries = []
    for threshold in answerability_thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("answerability thresholds must be between 0 and 1")
        predictions = [
            _outcome(
                float(row["decision_confidence"]),
                float(row["decision_conflict_probability"]),
                answerability_threshold=threshold,
                conflict_threshold=conflict_threshold,
            )
            for row in valid
        ]
        expected = [str(row["expected_outcome"]) for row in valid]
        answerable_predictions = [
            (prediction, target)
            for prediction, target in zip(predictions, expected)
            if prediction == "answerable"
        ]
        true_positive = sum(target == "answerable" for prediction, target in zip(predictions, expected) if prediction == "answerable")
        actual_answerable = sum(target == "answerable" for target in expected)
        summaries.append(
            {
                "answerability_threshold": threshold,
                "conflict_threshold": conflict_threshold,
                "evaluated_cases": len(valid),
                "end_to_end_outcome_accuracy": (
                    mean(prediction == target for prediction, target in zip(predictions, expected))
                    if valid
                    else None
                ),
                "answerable_precision": (
                    true_positive / len(answerable_predictions) if answerable_predictions else None
                ),
                "answerable_recall": true_positive / actual_answerable if actual_answerable else None,
                "false_answer_rate": (
                    sum(target != "answerable" for _prediction, target in answerable_predictions)
                    / len(answerable_predictions)
                    if answerable_predictions
                    else 0.0
                ),
                "abstention_rate": (
                    mean(prediction == "insufficient_evidence" for prediction in predictions)
                    if valid
                    else None
                ),
                "answer_coverage": (
                    mean(prediction == "answerable" for prediction in predictions) if valid else None
                ),
                "brier_score": (
                    mean(
                        (float(row["decision_confidence"]) - (1.0 if row["expected_outcome"] == "answerable" else 0.0)) ** 2
                        for row in valid
                    )
                    if valid
                    else None
                ),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Jev decision thresholds from recorded benchmark output")
    parser.add_argument("input", type=Path, help="JSON output from benchmarks.run")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--conflict-threshold", type=float, default=0.7)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    print(
        json.dumps(
            summarize_thresholds(
                payload["cases"],
                answerability_thresholds=args.thresholds,
                conflict_threshold=args.conflict_threshold,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
