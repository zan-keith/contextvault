from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, File, FastAPI, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db import Database
from app.extract import UnsupportedDocument, extract_text
from app.storage import FileStore


class FileCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=2000)
    content: str = Field(min_length=1)
    product: str | None = Field(default=None, max_length=200)
    version: str | None = Field(default=None, max_length=100)
    document_type: str = Field(min_length=1, max_length=100)
    status: str = Field(default="active", pattern="^(active|archived|superseded|restricted)$")


class QueryCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    product: str | None = Field(default=None, max_length=200)
    version: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


def create_app(database_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="ContextVault", version="0.1.0")
    db_path = database_path or os.getenv("CONTEXTVAULT_DB", "contextvault.db")
    app.state.database = Database(db_path)
    storage_root = os.getenv("CONTEXTVAULT_STORAGE_DIR")
    if storage_root is None:
        storage_root = str(Path(db_path).with_suffix(".files"))
    app.state.file_store = FileStore(storage_root)

    def database(request: Request) -> Database:
        return request.app.state.database

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/files", status_code=status.HTTP_201_CREATED)
    def create_file(payload: FileCreate, db: Database = Depends(database)) -> dict:
        return db.create_file(**payload.model_dump())

    @app.post("/files/upload", status_code=status.HTTP_201_CREATED)
    async def upload_file(
        request: Request,
        file: UploadFile = File(...),
        description: str = Form(...),
        product: str | None = Form(default=None),
        version: str | None = Form(default=None),
        document_type: str = Form(...),
        file_status: str = Form(default="active", alias="status"),
        db: Database = Depends(database),
    ) -> dict:
        allowed_extensions = {".txt", ".md", ".csv", ".json", ".pdf"}
        extension = Path(file.filename or "").suffix.lower()
        if extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only .txt, .md, .csv, .json, and .pdf files are supported in this slice",
            )

        raw_content = await file.read()
        try:
            content = extract_text(file.filename or "uploaded-file", raw_content)
        except UnsupportedDocument as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        stored = request.app.state.file_store.save(file.filename or "uploaded-file", raw_content)
        payload = FileCreate(
            name=file.filename or "uploaded-file",
            description=description,
            content=content,
            product=product,
            version=version,
            document_type=document_type,
            status=file_status,
        )
        return db.create_file(
            **payload.model_dump(),
            **stored,
            media_type=file.content_type,
        )

    @app.get("/files")
    def search_files(
        request: Request,
        q: str,
        product: str | None = None,
        version: str | None = None,
        limit: int = 10,
    ) -> dict:
        return {"query": q, "results": database(request).search(q, product=product, version=version, limit=limit)}

    @app.post("/queries")
    def query(payload: QueryCreate, db: Database = Depends(database)) -> dict:
        return {
            "query": payload.question,
            "results": db.search(
                payload.question,
                product=payload.product,
                version=payload.version,
                limit=payload.limit,
            ),
        }

    return app


app = create_app()
