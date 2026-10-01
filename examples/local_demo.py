"""Run a small ContextVault flow without starting a network server."""

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from app.main import create_app

with TemporaryDirectory(prefix="contextvault-demo-") as directory:
    client = TestClient(create_app(Path(directory) / "demo.db"))
    created = client.post(
        "/files",
        json={
            "name": "x200-maintenance.txt",
            "description": "X200 vacuum maintenance procedure",
            "content": "Inspect the vacuum sensor and seal before retrying calibration.",
            "product": "X200",
            "version": "B",
            "document_type": "troubleshooting",
            "status": "active",
        },
    )
    created.raise_for_status()

    response = client.post(
        "/queries",
        json={
            "question": "What should I inspect after X200 calibration fails?",
            "product": "X200",
            "version": "B",
            "strategy": "fts",
        },
    )
    response.raise_for_status()
    print(response.json())
