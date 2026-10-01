from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from fastembed import TextEmbedding

from app.db import Database


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class FastEmbedProvider:
    """CPU-friendly local embeddings powered by FastEmbed and ONNX Runtime."""

    name = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = name):
        self.name = model_name
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(list(texts))]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class HybridRetriever:
    """Combine deterministic metadata filters, FTS rank, and local embeddings.

    FTS5 supplies the lexical score; FastEmbed reranks all metadata-eligible
    chunks, which allows a semantic match even when the query uses no exact
    document wording. The returned scores are audit fields, not probabilities.
    """

    def __init__(
        self,
        database: Database,
        embedder: EmbeddingProvider,
        *,
        semantic_weight: float = 0.65,
        min_semantic_score: float = 0.65,
        candidate_limit: int = 1000,
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        if not 0 <= min_semantic_score <= 1:
            raise ValueError("min_semantic_score must be between 0 and 1")
        self.database = database
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.min_semantic_score = min_semantic_score
        self.candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        *,
        product: str | None = None,
        version: str | None = None,
        document_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        candidates = self.database.active_candidates(
            product=product,
            version=version,
            document_type=document_type,
            limit=self.candidate_limit,
        )
        if not candidates:
            return []

        lexical = self.database.search(
            query,
            product=product,
            version=version,
            document_type=document_type,
            limit=self.candidate_limit,
        )
        lexical_rank = {result["chunk_id"]: rank for rank, result in enumerate(lexical, start=1)}

        texts = [query, *[f"{candidate['description']}\n{candidate['text']}" for candidate in candidates]]
        try:
            vectors = self.embedder.embed(texts)
        except Exception:  # noqa: BLE001 - local FTS fallback must survive any embedding-provider failure
            fallback = []
            for result in lexical[:limit]:
                fallback.append(
                    {
                        **result,
                        "retrieval_strategy": "fts_fallback",
                        "embedding_model": self.embedder.name,
                        "lexical_score": 1 / (lexical.index(result) + 1),
                        "semantic_score": None,
                        "hybrid_score": None,
                    }
                )
            return fallback
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding provider returned an unexpected vector count")
        query_vector = vectors[0]

        results = []
        for candidate, vector in zip(candidates, vectors[1:]):
            semantic_score = max(0.0, _cosine(query_vector, vector))
            rank = lexical_rank.get(candidate["chunk_id"])
            lexical_score = 1 / rank if rank else 0.0
            if lexical_score == 0.0 and semantic_score < self.min_semantic_score:
                continue
            hybrid_score = (self.semantic_weight * semantic_score) + (
                (1 - self.semantic_weight) * lexical_score
            )
            results.append(
                {
                    **candidate,
                    "retrieval_strategy": "hybrid",
                    "embedding_model": self.embedder.name,
                    "lexical_score": round(lexical_score, 6),
                    "semantic_score": round(semantic_score, 6),
                    "hybrid_score": round(hybrid_score, 6),
                }
            )
        return sorted(results, key=lambda item: item["hybrid_score"], reverse=True)[:limit]
