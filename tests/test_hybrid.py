from app.db import Database
from app.retrieval import FastEmbedProvider, HybridRetriever, _cosine


def test_cosine_matches_known_vectors():
    assert _cosine([3.0, 4.0], [3.0, 4.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


class StaticEmbedder:
    name = "static"

    def embed(self, texts):
        vectors = {
            "How do I diagnose low pressure?": [1.0, 0.0],
            "Vacuum sensor inspection procedure\nVacuum sensor inspection procedure": [1.0, 0.0],
            "Install the network driver\nInstall the network driver": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


class WeakSemanticEmbedder:
    name = "weak-static"

    def embed(self, texts):
        return [[1.0, 0.0], *[[0.5, 0.866] for _ in texts[1:]]]


def test_fastembed_revision_is_part_of_persistent_cache_key(monkeypatch):
    monkeypatch.delenv("CONTEXTVAULT_EMBEDDING_MODEL_REVISION", raising=False)

    provider = FastEmbedProvider(model_revision="revision-a")

    assert provider.revision == "revision-a"
    assert provider.cache_key == "BAAI/bge-small-en-v1.5@revision-a"


def test_hybrid_retrieval_can_rank_semantic_match_without_lexical_overlap(tmp_path):
    db = Database(tmp_path / "hybrid.db")
    db.create_file(
        name="vacuum.txt",
        description="Vacuum sensor inspection procedure",
        content="Vacuum sensor inspection procedure",
        product="X200",
        version="B",
        document_type="troubleshooting",
        status="active",
    )
    db.create_file(
        name="network.txt",
        description="Install the network driver",
        content="Install the network driver",
        product="X200",
        version="B",
        document_type="manual",
        status="active",
    )

    results = HybridRetriever(db, StaticEmbedder(), min_semantic_score=0.0).search(
        "How do I diagnose low pressure?", product="X200", version="B"
    )

    assert results[0]["name"] == "vacuum.txt"
    assert results[0]["semantic_score"] == 1.0
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


class CountingEmbedder:
    name = "counting"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]


def test_hybrid_reuses_persisted_document_embeddings(tmp_path):
    db = Database(tmp_path / "embedding-cache.db")
    db.create_file(
        name="vacuum.txt",
        description="Vacuum sensor inspection procedure",
        content="Vacuum sensor inspection procedure",
        product="X200",
        version="B",
        document_type="troubleshooting",
        status="active",
    )
    embedder = CountingEmbedder()
    retriever = HybridRetriever(db, embedder, min_semantic_score=0.0)

    retriever.search("How do I diagnose low pressure?", product="X200")
    retriever.search("How do I diagnose low pressure?", product="X200")

    assert embedder.calls == [2, 1]


def test_hybrid_retrieval_respects_metadata_filters(tmp_path):
    db = Database(tmp_path / "hybrid-filter.db")
    db.create_file(
        name="x200.txt",
        description="Vacuum sensor inspection procedure",
        content="Vacuum sensor inspection procedure",
        product="X200",
        version="B",
        document_type="troubleshooting",
        status="active",
    )
    db.create_file(
        name="x300.txt",
        description="Vacuum sensor inspection procedure",
        content="Vacuum sensor inspection procedure",
        product="X300",
        version="B",
        document_type="troubleshooting",
        status="active",
    )

    results = HybridRetriever(db, StaticEmbedder(), min_semantic_score=0.0).search(
        "How do I diagnose low pressure?", product="X200"
    )

    assert [result["name"] for result in results] == ["x200.txt"]


def test_hybrid_retrieval_abstains_from_weak_semantic_only_match(tmp_path):
    db = Database(tmp_path / "hybrid-threshold.db")
    db.create_file(
        name="unrelated.txt",
        description="Unrelated maintenance procedure",
        content="A generic maintenance instruction.",
        product="X200",
        version="B",
        document_type="manual",
        status="active",
    )

    results = HybridRetriever(db, WeakSemanticEmbedder()).search("quantum reactor repair")

    assert results == []
