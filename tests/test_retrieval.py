import sqlite3

from app.db import Database


def test_search_returns_relevant_active_file(tmp_path):
    db = Database(tmp_path / "test.db")
    db.create_file(
        name="x200-calibration.pdf",
        description="Calibration troubleshooting for the X200 handler",
        content="If calibration fails, inspect the vacuum sensor before rerunning calibration.",
        product="X200",
        version="B",
        document_type="troubleshooting",
        status="active",
    )
    db.create_file(
        name="x200-old.pdf",
        description="Old calibration instructions",
        content="Calibration instructions for the previous revision.",
        product="X200",
        version="A",
        document_type="manual",
        status="superseded",
    )

    results = db.search("calibration vacuum sensor", product="X200", version="B")

    assert len(results) == 1
    assert results[0]["name"] == "x200-calibration.pdf"
    assert results[0]["status"] == "active"
    assert "vacuum sensor" in results[0]["text"]


def test_search_does_not_return_other_products(tmp_path):
    db = Database(tmp_path / "test.db")
    db.create_file(
        name="x200.pdf",
        description="X200 calibration manual",
        content="X200 calibration procedure",
        product="X200",
        version="B",
        document_type="manual",
        status="active",
    )
    db.create_file(
        name="x300.pdf",
        description="X300 calibration manual",
        content="X300 calibration procedure",
        product="X300",
        version="B",
        document_type="manual",
        status="active",
    )

    results = db.search("calibration", product="X200")

    assert [result["name"] for result in results] == ["x200.pdf"]




def test_search_can_filter_document_type(tmp_path):
    db = Database(tmp_path / "types.db")
    db.create_file(
        name="x200-manual.pdf",
        description="X200 installation manual",
        content="Install the X200 module before calibration.",
        product="X200",
        version="B",
        document_type="manual",
        status="active",
    )
    db.create_file(
        name="x200-troubleshooting.pdf",
        description="X200 calibration troubleshooting",
        content="If calibration fails, inspect the vacuum sensor.",
        product="X200",
        version="B",
        document_type="troubleshooting",
        status="active",
    )

    results = db.search("calibration", product="X200", document_type="troubleshooting")

    assert [result["name"] for result in results] == ["x200-troubleshooting.pdf"]


def test_search_ignores_common_question_words_for_no_answer(tmp_path):
    db = Database(tmp_path / "test.db")
    db.create_file(
        name="deployment.txt",
        description="The deployment procedure",
        content="The team deploys the service after review.",
        product="web",
        version="v1",
        document_type="runbook",
        status="active",
    )

    assert db.search("What is the recipe for sourdough bread?") == []


def test_database_schema_is_initialised(tmp_path):
    db = Database(tmp_path / "test.db")
    with sqlite3.connect(db.path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
            )
        }

    assert {"files", "chunks", "chunks_fts"}.issubset(tables)
