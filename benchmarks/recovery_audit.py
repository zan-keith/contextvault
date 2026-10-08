from __future__ import annotations

import json
from pathlib import Path

from app.db import Database
from app.recovery import DatabaseRecoverySearch, _merge_candidates, build_recovery_queries

ROOT = Path(__file__).parent


def candidate_names(db: Database, question: str, query: dict, limit: int = 5) -> list[str]:
    rows = db.search(
        question,
        product=query.get("product"),
        version=query.get("version"),
        document_type=query.get("document_type"),
        limit=limit,
    )
    return [row["name"] for row in rows]


def recovery_adds_nothing_new(
    db: Database,
    question: str,
    query: dict,
    *,
    limit: int = 5,
    recovery_limit: int = 10,
) -> bool:
    """True when every recovery query only re-returns already-seen chunks."""
    initial = db.search(
        question,
        product=query.get("product"),
        version=query.get("version"),
        document_type=query.get("document_type"),
        limit=limit,
    )
    merged = list(initial)
    for recovery_query in build_recovery_queries(question):
        recovered = db.search(
            recovery_query,
            product=query.get("product"),
            version=query.get("version"),
            document_type=query.get("document_type"),
            limit=recovery_limit,
        )
        merged = _merge_candidates(merged, recovered, limit=limit)
    return len(merged) == len(initial)


def main() -> None:
    import tempfile

    corpus = json.loads((ROOT / "robust-text-corpus.json").read_text())
    queries = json.loads((ROOT / "robust-text-queries.json").read_text())
    with tempfile.TemporaryDirectory(prefix="contextvault-recovery-audit-") as directory:
        db = Database(Path(directory) / "audit.db")
        db.create_files(corpus)
        report = []
        for query in queries:
            question = query["question"]
            initial = candidate_names(db, question, query)
            recovery_search = DatabaseRecoverySearch(db)
            merged = list(
                db.search(
                    question,
                    product=query.get("product"),
                    version=query.get("version"),
                    document_type=query.get("document_type"),
                    limit=5,
                )
            )
            for recovery_query in build_recovery_queries(question):
                recovered = recovery_search.search(
                    recovery_query,
                    product=query.get("product"),
                    version=query.get("version"),
                    document_type=query.get("document_type"),
                    limit=10,
                )
                merged = _merge_candidates(merged, recovered, limit=5)
            report.append(
                {
                    "case_id": query.get("id"),
                    "expected_files": sorted(query.get("expected_files", [])),
                    "initial_files": initial,
                    "recovery_final_files": [candidate["name"] for candidate in merged],
                    "recovery_added_new": len(merged) > len(initial),
                    "all_expected_reachable": set(query.get("expected_files", [])) <= {c["name"] for c in merged},
                }
            )
    unreachable = [row for row in report if not row["all_expected_reachable"]]
    print(json.dumps({"cases": len(report), "cases_with_unreachable_expected_files": unreachable}, indent=2))


if __name__ == "__main__":
    main()
