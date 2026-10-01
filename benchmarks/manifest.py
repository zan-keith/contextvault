from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BENCHMARK_VERSION = "contextvault-benchmark-v2"


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_version: str
    git_commit: str | None
    corpus_sha256: str
    query_set_sha256: str
    embedding_model: str | None
    embedding_model_revision: str | None
    reranker_revision: str | None
    decision_model_revision: str | None
    retrieval_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def build_manifest(
    corpus_path: Path,
    queries_path: Path,
    *,
    repository: Path,
    retrieval_config: dict[str, Any],
    embedding_model: str | None = None,
    embedding_model_revision: str | None = None,
    reranker_revision: str | None = None,
    decision_model_revision: str | None = None,
) -> BenchmarkManifest:
    return BenchmarkManifest(
        benchmark_version=BENCHMARK_VERSION,
        git_commit=git_commit(repository),
        corpus_sha256=sha256_file(corpus_path),
        query_set_sha256=sha256_file(queries_path),
        embedding_model=embedding_model,
        embedding_model_revision=embedding_model_revision,
        reranker_revision=reranker_revision,
        decision_model_revision=decision_model_revision,
        retrieval_config=retrieval_config,
    )
