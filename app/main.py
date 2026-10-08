from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.answers import (
    GenerationProvider,
    GenerationProviderError,
    GenerationUnavailableError,
    build_evidence_manifest,
    generation_provider_from_environment,
    validate_generated_answer,
)
from app.db import Database
from app.decisions import DecisionProvider, provider_from_environment
from app.extract import UnsupportedDocument, extract_text
from app.recovery import DatabaseRecoverySearch, answer_with_recovery
from app.retrieval import EmbeddingProvider, FastEmbedProvider, HybridRetriever
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
    document_type: str | None = Field(default=None, max_length=100)
    strategy: Literal["fts", "hybrid"] = "hybrid"
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class RecoveryQueryCreate(QueryCreate):
    max_rounds: int = Field(default=2, ge=1, le=5)


def create_app(
    database_path: str | Path | None = None,
    decision_provider: DecisionProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    generation_provider: GenerationProvider | None = None,
) -> FastAPI:
    app = FastAPI(title="ContextVault", version="0.1.0")
    db_path = database_path or os.getenv("CONTEXTVAULT_DB", "contextvault.db")
    app.state.database = Database(db_path)
    storage_root = os.getenv("CONTEXTVAULT_STORAGE_DIR")
    if storage_root is None:
        storage_root = str(Path(db_path).with_suffix(".files"))
    app.state.file_store = FileStore(storage_root)
    app.state.decision_provider = decision_provider or provider_from_environment()
    app.state.generation_provider = generation_provider or generation_provider_from_environment()
    try:
        app.state.max_upload_bytes = int(os.getenv("CONTEXTVAULT_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    except ValueError as exc:
        raise ValueError("CONTEXTVAULT_MAX_UPLOAD_BYTES must be an integer") from exc
    if app.state.max_upload_bytes < 1:
        raise ValueError("CONTEXTVAULT_MAX_UPLOAD_BYTES must be positive")
    embedder = embedding_provider or FastEmbedProvider()
    app.state.hybrid_retriever = HybridRetriever(app.state.database, embedder)
    app.state.evidence_retriever = HybridRetriever(
        app.state.database,
        embedder,
        min_semantic_score=0.0,
    )

    def database(request: Request) -> Database:
        return request.app.state.database

    def get_decision_provider(request: Request) -> DecisionProvider:
        return request.app.state.decision_provider

    def hybrid_retriever(request: Request) -> HybridRetriever:
        return request.app.state.hybrid_retriever

    def evidence_retriever(request: Request) -> HybridRetriever:
        return request.app.state.evidence_retriever

    def get_generation_provider(request: Request) -> GenerationProvider:
        return request.app.state.generation_provider

    def public_file(record: dict) -> dict:
        """Do not expose server-local content-addressed storage paths via the API."""
        return {key: value for key, value in record.items() if key != "source_uri"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/files", status_code=status.HTTP_201_CREATED)
    def create_file(payload: FileCreate, db: Database = Depends(database)) -> dict:
        return public_file(db.create_file(**payload.model_dump()))

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

        content_length = request.headers.get("content-length")
        try:
            declared_size = int(content_length) if content_length is not None else None
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Length header") from exc
        if declared_size is not None and declared_size > request.app.state.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Upload exceeds the maximum size of {request.app.state.max_upload_bytes} bytes",
            )
        raw_content = await file.read()
        if len(raw_content) > request.app.state.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Upload exceeds the maximum size of {request.app.state.max_upload_bytes} bytes",
            )
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
        return public_file(
            db.create_file(
                **payload.model_dump(),
                **stored,
                media_type=file.content_type,
            )
        )

    @app.get("/files")
    def search_files(
        request: Request,
        q: str,
        product: str | None = None,
        version: str | None = None,
        document_type: str | None = None,
        strategy: Literal["fts", "hybrid"] = "hybrid",
        limit: int = 10,
        retriever: HybridRetriever = Depends(hybrid_retriever),
    ) -> dict:
        if strategy == "hybrid":
            results = retriever.search(
                q,
                product=product,
                version=version,
                document_type=document_type,
                limit=limit,
            )
        else:
            results = database(request).search(
                q,
                product=product,
                version=version,
                document_type=document_type,
                limit=limit,
            )
        return {"query": q, "strategy": strategy, "results": results}

    @app.post("/queries")
    async def query(
        payload: QueryCreate,
        db: Database = Depends(database),
        provider: DecisionProvider = Depends(get_decision_provider),
        evidence: HybridRetriever = Depends(evidence_retriever),
    ) -> dict:
        if payload.strategy == "hybrid":
            results = evidence.search(
                payload.question,
                product=payload.product,
                version=payload.version,
                document_type=payload.document_type,
                limit=payload.limit,
            )
        else:
            results = db.search(
                payload.question,
                product=payload.product,
                version=payload.version,
                document_type=payload.document_type,
                limit=payload.limit,
            )
        decision = await provider.evaluate(payload.question, results)
        visible_results = results if decision.outcome == "answerable" else []
        return {
            "query": payload.question,
            "strategy": payload.strategy,
            "decision": {
                "outcome": decision.outcome,
                "confidence": decision.confidence,
                "support_probability": decision.support_probability,
                "conflict_probability": decision.conflict_probability,
                "coverage": decision.coverage,
                "applicability": decision.applicability,
                "freshness": decision.freshness,
                "provider": decision.provider,
                "provider_version": decision.provider_version,
                "reasons": decision.reasons,
            },
            "results": visible_results,
        }

    @app.post("/answers")
    async def answer(
        payload: QueryCreate,
        provider: DecisionProvider = Depends(get_decision_provider),
        generator: GenerationProvider = Depends(get_generation_provider),
        evidence: HybridRetriever = Depends(evidence_retriever),
        db: Database = Depends(database),
    ) -> dict:
        if payload.strategy == "hybrid":
            candidates = evidence.search(
                payload.question,
                product=payload.product,
                version=payload.version,
                document_type=payload.document_type,
                limit=payload.limit,
            )
        else:
            candidates = db.search(
                payload.question,
                product=payload.product,
                version=payload.version,
                document_type=payload.document_type,
                limit=payload.limit,
            )

        try:
            decision = await provider.evaluate(payload.question, candidates)
        except Exception:  # noqa: BLE001 - answer generation must fail closed
            return {
                "query": payload.question,
                "answer": None,
                "abstention": {
                    "reason": "decision_provider_error",
                    "action": "Review the evidence or try again later.",
                },
            }

        outcome = getattr(decision, "outcome", None)
        trusted = getattr(decision, "trusted", False)
        degraded = getattr(decision, "degraded", False)
        conflict_probability = getattr(decision, "conflict_probability", None)
        if (
            not candidates
            or outcome != "answerable"
            or not trusted
            or degraded
            or (conflict_probability is not None and conflict_probability >= 0.7)
        ):
            reason = "empty_evidence" if not candidates else str(outcome or "invalid_decision")
            if not trusted and outcome == "answerable":
                reason = "untrusted_decision"
            elif degraded:
                reason = "degraded_decision"
            elif conflict_probability is not None and conflict_probability >= 0.7:
                reason = "conflicting_evidence"
            return {
                "query": payload.question,
                "answer": None,
                "abstention": {
                    "reason": reason,
                    "action": "Review the evidence before relying on an answer.",
                },
            }

        manifest = build_evidence_manifest(candidates)
        try:
            generated = await generator.generate(payload.question, manifest)
            validated = validate_generated_answer(generated, manifest)
        except GenerationUnavailableError:
            return {
                "query": payload.question,
                "answer": None,
                "abstention": {
                    "reason": "generation_provider_unavailable",
                    "action": "Configure an answer provider or review the evidence manually.",
                },
            }
        except GenerationProviderError:
            return {
                "query": payload.question,
                "answer": None,
                "abstention": {
                    "reason": "generation_provider_error",
                    "action": "The answer provider failed; review the evidence or try again later.",
                },
            }
        except ValueError:
            return {
                "query": payload.question,
                "answer": None,
                "abstention": {
                    "reason": "invalid_generated_answer",
                    "action": "Review the evidence and answer manually.",
                },
            }
        except Exception:  # noqa: BLE001 - unexpected generation errors still fail closed
            return {
                "query": payload.question,
                "answer": None,
                "abstention": {
                    "reason": "invalid_generated_answer",
                    "action": "Review the evidence and answer manually.",
                },
            }
        return {
            "query": payload.question,
            "answer": {"text": validated.answer, "claims": [claim.model_dump() for claim in validated.claims]},
            "generation": {
                "provider": validated.provider,
                "provider_version": validated.provider_version,
                "latency_ms": validated.latency_ms,
                "input_tokens": validated.input_tokens,
                "output_tokens": validated.output_tokens,
                "cost_usd": validated.cost_usd,
            },
            "abstention": None,
        }

    @app.post("/answers/recovery")
    async def answer_with_recovery_endpoint(
        payload: RecoveryQueryCreate,
        provider: DecisionProvider = Depends(get_decision_provider),
        generator: GenerationProvider = Depends(get_generation_provider),
        evidence: HybridRetriever = Depends(evidence_retriever),
        db: Database = Depends(database),
    ) -> dict:
        """Jev-gated answers with targeted recovery search after a rejection.

        Unlike ``POST /answers`` this endpoint may re-run the decision gate after
        adding recovered candidates, so a single missing fact becomes a targeted
        search instead of an abstention. Generation is still gated on a trusted,
        non-degraded, conflict-free answerable decision and only ever sees the
        server-built evidence manifest.
        """
        if payload.strategy == "hybrid":
            candidates = evidence.search(
                payload.question,
                product=payload.product,
                version=payload.version,
                document_type=payload.document_type,
                limit=payload.limit,
            )
        else:
            candidates = db.search(
                payload.question,
                product=payload.product,
                version=payload.version,
                document_type=payload.document_type,
                limit=payload.limit,
            )
        return await answer_with_recovery(
            payload.question,
            candidates,
            decision_provider=provider,
            generation_provider=generator,
            recovery_search=DatabaseRecoverySearch(db),
            product=payload.product,
            version=payload.version,
            document_type=payload.document_type,
            limit=payload.limit,
            max_rounds=payload.max_rounds,
        )

    return app


app = create_app()
