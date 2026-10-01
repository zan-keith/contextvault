from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


class Database:
    """Small SQLite-backed repository for the first retrieval baseline."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
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
        chunks = self._chunks(content)
        with self._connect() as connection:
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

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", query.lower())
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
                bm25(chunks_fts) AS search_score
            FROM chunks_fts AS fts
            JOIN files ON files.id = fts.file_id
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
                files.document_type AS document_type
            FROM chunks
            JOIN files ON files.id = chunks.file_id
            WHERE {' AND '.join(clauses)}
            ORDER BY files.id, chunks.ordinal
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]
