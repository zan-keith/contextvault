from typing import Any

from fastapi.testclient import TestClient

from app.decisions import DecisionResult
from app.main import create_app


class ParaphraseEmbedder:
    name = "test-paraphrase-embedder"

    def embed(self, texts):
        vectors = []
        for text in texts:
            if text.startswith("How do I renew encryption"):
                vectors.append([1.0, 0.0])
            elif "TLS certificate" in text:
                vectors.append([0.62, 0.7846018098373212])
            else:
                vectors.append([0.1, 0.99498743710662])
        return vectors


class AcceptingDecisionProvider:
    async def evaluate(self, _question: str, candidates: list[dict[str, Any]]) -> DecisionResult:
        return DecisionResult(
            outcome="answerable" if candidates else "insufficient_evidence",
            confidence=1.0 if candidates else 0.0,
            reasons=["test decision"],
            provider="jev",
        )


class RejectingDecisionProvider:
    async def evaluate(self, _question: str, _candidates: list[dict[str, Any]]) -> DecisionResult:
        return DecisionResult(
            outcome="insufficient_evidence",
            confidence=1.0,
            reasons=["test rejection"],
            provider="jev",
        )


def test_upload_response_does_not_expose_internal_storage_path(tmp_path):
    client = TestClient(create_app(tmp_path / "api.db"))

    response = client.post(
        "/files/upload",
        files={"file": ("manual.txt", b"Inspect the vacuum sensor.", "text/plain")},
        data={"description": "Manual", "document_type": "maintenance"},
    )

    assert response.status_code == 201
    assert response.json()["sha256"]
    assert "source_uri" not in response.json()


def test_upload_rejects_files_larger_than_configured_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXTVAULT_MAX_UPLOAD_BYTES", "10")
    client = TestClient(create_app(tmp_path / "api.db"))

    response = client.post(
        "/files/upload",
        files={"file": ("manual.txt", b"12345678901", "text/plain")},
        data={"description": "Manual", "document_type": "maintenance"},
    )

    assert response.status_code == 413
    assert "maximum size" in response.json()["detail"]


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
    assert body["decision"]["provider"] == "rules"
    assert body["decision"]["outcome"] == "answerable"




def test_hybrid_query_sends_broader_semantic_candidates_to_decision_provider(tmp_path):
    client = TestClient(
        create_app(
            tmp_path / "semantic.db",
            decision_provider=AcceptingDecisionProvider(),
            embedding_provider=ParaphraseEmbedder(),
        )
    )
    client.post(
        "/files",
        json={
            "name": "web-tls-deployment.txt",
            "description": "Procedure for deploying a TLS certificate.",
            "content": "Obtain the TLS certificate and install the certificate chain.",
            "product": "web",
            "version": "v2",
            "document_type": "runbook",
            "status": "active",
        },
    )

    response = client.post(
        "/queries",
        json={
            "question": "How do I renew encryption credentials for the public website?",
            "product": "web",
            "version": "v2",
            "document_type": "runbook",
            "strategy": "hybrid",
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"]["outcome"] == "answerable"
    assert response.json()["results"][0]["name"] == "web-tls-deployment.txt"


def test_query_hides_evidence_rejected_by_decision_provider(tmp_path):
    client = TestClient(
        create_app(tmp_path / "rejected.db", decision_provider=RejectingDecisionProvider())
    )
    client.post(
        "/files",
        json={
            "name": "payment-runbook.txt",
            "description": "Payment outage procedure",
            "content": "Queue failed charges until the payment processor is healthy.",
            "product": "billing",
            "version": "v1",
            "document_type": "runbook",
            "status": "active",
        },
    )

    response = client.post(
        "/queries",
        json={"question": "How should failed charges be handled?", "strategy": "fts"},
    )

    assert response.status_code == 200
    assert response.json()["decision"]["outcome"] == "insufficient_evidence"
    assert response.json()["results"] == []


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
    body = response.json()
    assert body["name"] == "x200-calibration.txt"
    stored = next((tmp_path / "upload.files").glob("*.txt"))
    assert stored.read_bytes() == b"If calibration fails, inspect the vacuum sensor before retrying."

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


def _text_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return pdf


def test_text_pdf_upload_is_extracted_and_stored(tmp_path):
    client = TestClient(create_app(tmp_path / "pdf.db"))

    response = client.post(
        "/files/upload",
        data={"description": "X200 calibration manual", "product": "X200", "document_type": "manual"},
        files={"file": ("manual.pdf", _text_pdf("Calibration procedure for X200"), "application/pdf")},
    )

    assert response.status_code == 201
    stored = next((tmp_path / "pdf.files").glob("*.pdf"))
    assert stored.read_bytes().startswith(b"%PDF-1.4")

    response = client.post("/queries", json={"question": "What is the calibration procedure?"})

    assert response.status_code == 200
    assert "Calibration procedure" in response.json()["results"][0]["text"]


def test_pdf_without_text_is_rejected_for_future_ocr(tmp_path):
    client = TestClient(create_app(tmp_path / "scanned.db"))

    response = client.post(
        "/files/upload",
        data={"description": "Scanned manual", "document_type": "manual"},
        files={"file": ("scanned.pdf", _text_pdf(""), "application/pdf")},
    )

    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]
