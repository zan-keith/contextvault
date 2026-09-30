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




def test_text_file_upload_and_query(tmp_path):
    client = TestClient(create_app(tmp_path / "upload.db"))

    response = client.post(
        "/files/upload",
        data={
            "description": "Calibration troubleshooting for the X200 module",
            "product": "X200",
            "version": "B",
            "document_type": "troubleshooting",
            "status": "active",
        },
        files={
            "file": (
                "x200-calibration.txt",
                b"If calibration fails, inspect the vacuum sensor before retrying.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["name"] == "x200-calibration.txt"

    response = client.post(
        "/queries",
        json={"question": "What should I inspect after calibration fails?", "product": "X200"},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["name"] == "x200-calibration.txt"


def test_query_requires_non_empty_question(tmp_path):
    client = TestClient(create_app(tmp_path / "api.db"))

    response = client.post("/queries", json={"question": "   "})

    assert response.status_code == 422


def test_upload_rejects_pdf_until_extraction_is_implemented(tmp_path):
    client = TestClient(create_app(tmp_path / "pdf.db"))

    response = client.post(
        "/files/upload",
        data={"description": "A PDF manual", "document_type": "manual"},
        files={"file": ("manual.pdf", b"%PDF-1.7", "application/pdf")},
    )

    assert response.status_code == 415
