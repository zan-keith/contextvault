from fastapi.testclient import TestClient

from app.main import create_app


def test_file_ingestion_and_query_round_trip(tmp_path):
    client = TestClient(create_app(tmp_path / "api.db"))

    response = client.post(
        "/files",
        json={
            "name": "x200-maintenance.txt",
            "description": "Maintenance procedure for the X200 module",
            "content": "For a vacuum alarm, inspect the vacuum sensor and check the seal.",
            "product": "X200",
            "version": "B",
            "document_type": "maintenance",
            "status": "active",
        },
    )

    assert response.status_code == 201
    assert response.json()["name"] == "x200-maintenance.txt"

    response = client.post(
        "/queries",
        json={"question": "How do I investigate a vacuum alarm?", "product": "X200", "version": "B"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "How do I investigate a vacuum alarm?"
    assert body["results"][0]["name"] == "x200-maintenance.txt"
    assert body["results"][0]["product"] == "X200"


def test_query_requires_non_empty_question(tmp_path):
    client = TestClient(create_app(tmp_path / "api.db"))

    response = client.post("/queries", json={"question": "   "})

    assert response.status_code == 422
