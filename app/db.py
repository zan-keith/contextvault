from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any


class Database:
    """Small SQLite-backed repository for the first retrieval baseline."""

    _busy_timeout_ms = 30_000

    _fts_stopwords = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "can",
            "do",
            "for",
            "how",
            "i",
            "in",
            "is",
            "of",
            "the",
            "to",
            "what",
            "when",
            "where",
            "why",
            "with",
        }
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self._busy_timeout_ms / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return connection

    def _initialise(self) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    product TEXT,
                    version TEXT,
                    document_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_uri TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    media_type TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chunk_id, model)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    file_id UNINDEXED,
                    chunk_id UNINDEXED,
                    name,
                    description,
                    text,
                    product UNINDEXED,
                    version UNINDEXED,
                    status UNINDEXED
                );
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(files)")}
            for column, definition in {
                "source_uri": "TEXT",
                "sha256": "TEXT",
                "size_bytes": "INTEGER",
                "media_type": "TEXT",
            }.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE files ADD COLUMN {column} {definition}")

    @staticmethod
    def _chunks(content: str, max_chars: int = 900) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        chunks: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= max_chars:
                chunks.append(paragraph)
                continue
            chunks.extend(
                paragraph[start : start + max_chars]
                for start in range(0, len(paragraph), max_chars)
            )
        return chunks or [content.strip()]

    def create_file(
        self,
        *,
        name: str,
        description: str,
        content: str,
        product: str | None,
        version: str | None,
        document_type: str,
        status: str,
        source_uri: str | None = None,
        sha256: str | None = None,
        size_bytes: int | None = None,
        media_type: str | None = None,
    ) -> dict[str, Any]:
        if sha256 is None:
            sha256 = hashlib.sha256(content.encode()).hexdigest()
        chunks = self._chunks(content)
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO files
                    (name, description, product, version, document_type, status,
                     source_uri, sha256, size_bytes, media_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    description,
                    product,
                    version,
                    document_type,
                    status,
                    source_uri,
                    sha256,
                    size_bytes,
                    media_type,
                ),
            )
            file_id = cursor.lastrowid
            for ordinal, text in enumerate(chunks):
                chunk_cursor = connection.execute(
                    "INSERT INTO chunks (file_id, ordinal, text) VALUES (?, ?, ?)",
                    (file_id, ordinal, text),
                )
                connection.execute(
                    """
                    INSERT INTO chunks_fts
                        (file_id, chunk_id, name, description, text, product, version, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        chunk_cursor.lastrowid,
                        name,
                        description,
                        text,
                        product or "",
                        version or "",
                        status,
                    ),
                )
            row = connection.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row)

    def create_files(self, documents: list[dict[str, Any]]) -> None:
        """Insert multiple files in one transaction for local bulk imports."""
        with self._write_lock, self._connect() as connection:
            for document in documents:
                content = document["content"]
                document_sha256 = document.get("sha256") or hashlib.sha256(content.encode()).hexdigest()
                chunks = self._chunks(content)
                cursor = connection.execute(
                    """
                    INSERT INTO files
                        (name, description, product, version, document_type, status,
                         source_uri, sha256, size_bytes, media_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document["name"],
                        document["description"],
                        document.get("product"),
                        document.get("version"),
                        document["document_type"],
                        document["status"],
                        document.get("source_uri"),
                        document_sha256,
                        document.get("size_bytes"),
                        document.get("media_type"),
                    ),
                )
                file_id = cursor.lastrowid
                for ordinal, text in enumerate(chunks):
                    chunk_cursor = connection.execute(
                        "INSERT INTO chunks (file_id, ordinal, text) VALUES (?, ?, ?)",
                        (file_id, ordinal, text),
                    )
                    connection.execute(
                        """
                        INSERT INTO chunks_fts
                            (file_id, chunk_id, name, description, text, product, version, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            file_id,
                            chunk_cursor.lastrowid,
                            document["name"],
                            document["description"],
                            text,
                            document.get("product") or "",
                            document.get("version") or "",
                            document["status"],
                        ),
                    )

    @classmethod
    def _fts_query(cls, query: str) -> str:
        tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9_]+", query.lower())
            if token not in cls._fts_stopwords
        ]
        return " OR ".join(f'"{token}"' for token in tokens)

    def search(
        self,
        query: str,
        *,
        product: str | None = None,
        version: str | None = None,
        document_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        fts_query = self._fts_query(query)
        if not fts_query:
            return []

        clauses = ["chunks_fts MATCH ?", "fts.status = 'active'"]
        parameters: list[Any] = [fts_query]
        if product:
            clauses.append("fts.product = ?")
            parameters.append(product)
        if version:
            clauses.append("fts.version = ?")
            parameters.append(version)
        if document_type:
            clauses.append("files.document_type = ?")
            parameters.append(document_type)
        parameters.append(max(1, min(limit, 50)))

        sql = f"""
            SELECT
                fts.file_id AS file_id,
                fts.chunk_id AS chunk_id,
                fts.name AS name,
                fts.description AS description,
                fts.text AS text,
                fts.product AS product,
                fts.version AS version,
                fts.status AS status,
                files.document_type AS document_type,
                files.sha256 AS sha256,
                chunks.ordinal AS ordinal,
                bm25(chunks_fts) AS search_score
            FROM chunks_fts AS fts
            JOIN files ON files.id = fts.file_id
            JOIN chunks ON chunks.id = fts.chunk_id
            WHERE {' AND '.join(clauses)}
            ORDER BY search_score ASC
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def active_candidates(
        self,
        *,
        product: str | None = None,
        version: str | None = None,
        document_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses = ["files.status = 'active'"]
        parameters: list[Any] = []
        if product:
            clauses.append("files.product = ?")
            parameters.append(product)
        if version:
            clauses.append("files.version = ?")
            parameters.append(version)
        if document_type:
            clauses.append("files.document_type = ?")
            parameters.append(document_type)
        parameters.append(max(1, min(limit, 5000)))
        sql = f"""
            SELECT
                files.id AS file_id,
                chunks.id AS chunk_id,
                files.name AS name,
                files.description AS description,
                chunks.text AS text,
                files.product AS product,
                files.version AS version,
                files.status AS status,
                files.document_type AS document_type,
                files.sha256 AS sha256,
                chunks.ordinal AS ordinal
            FROM chunks
            JOIN files ON files.id = chunks.file_id
            WHERE {' AND '.join(clauses)}
            ORDER BY files.id, chunks.ordinal
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def get_chunk_embeddings(self, chunk_ids: list[int], model: str) -> dict[int, list[float]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT chunk_id, vector_json FROM chunk_embeddings WHERE model = ? AND chunk_id IN ({placeholders})",
                [model, *chunk_ids],
            ).fetchall()
        return {int(row["chunk_id"]): json.loads(row["vector_json"]) for row in rows}

    def upsert_chunk_embeddings(
        self,
        embeddings: list[tuple[int, str, list[float]]],
    ) -> None:
        if not embeddings:
            return
        with self._write_lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunk_embeddings (chunk_id, model, vector_json)
                VALUES (?, ?, ?)
                ON CONFLICT(chunk_id, model) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                [
                    (chunk_id, model, json.dumps(vector, separators=(",", ":")))
                    for chunk_id, model, vector in embeddings
                ],
            )
