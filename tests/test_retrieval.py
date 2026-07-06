"""Unit tests for rag/retrieval.py - pure logic only, no FAISS/embeddings/network."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.retrieval import build_queries, retrieve


def _enriched_stats(overrides: dict | None = None) -> dict:
    base = {
        "total": 224 * 224,
        0: {"pixels": 30000, "pct": 60.0, "mean_confidence": 0.95, "mean_entropy": 0.1, "low_confidence": False},
        1: {"pixels": 10000, "pct": 20.0, "mean_confidence": 0.9, "mean_entropy": 0.2, "low_confidence": False},
        2: {"pixels": 3000, "pct": 6.0, "mean_confidence": 0.85, "mean_entropy": 0.3, "low_confidence": False},
        3: {"pixels": 3000, "pct": 6.0, "mean_confidence": 0.85, "mean_entropy": 0.3, "low_confidence": False},
        4: {"pixels": 4000, "pct": 8.0, "mean_confidence": 0.8, "mean_entropy": 0.3, "low_confidence": False},
    }
    for cls_id, patch in (overrides or {}).items():
        base[cls_id] = {**base[cls_id], **patch}
    return base


def test_build_queries_returns_2_to_4_queries():
    queries = build_queries(_enriched_stats())
    assert 2 <= len(queries) <= 4
    assert all(isinstance(q, str) and q for q in queries)


def test_build_queries_deterministic():
    stats = _enriched_stats({1: {"low_confidence": True}})
    first = build_queries(stats)
    second = build_queries(stats)
    assert first == second


def test_build_queries_flags_uncertainty_query():
    stats = _enriched_stats({1: {"low_confidence": True}})
    queries = build_queries(stats)
    assert any("uncertain" in q.lower() or "confidence" in q.lower() or "review" in q.lower() for q in queries)


def test_build_queries_flags_kidney_asymmetry():
    stats = _enriched_stats({2: {"pct": 10.0}, 3: {"pct": 2.0}})
    queries = build_queries(stats)
    assert any("asymmetry" in q.lower() for q in queries)


def test_build_queries_no_asymmetry_within_threshold():
    stats = _enriched_stats({2: {"pct": 6.0}, 3: {"pct": 5.5}})
    queries = build_queries(stats)
    assert not any("asymmetry" in q.lower() for q in queries)


def test_build_queries_handles_missing_confidence_keys():
    legacy_stats = {
        "total": 224 * 224,
        0: {"pixels": 30000, "pct": 60.0},
        1: {"pixels": 10000, "pct": 20.0},
        2: {"pixels": 3000, "pct": 6.0},
        3: {"pixels": 3000, "pct": 6.0},
        4: {"pixels": 4000, "pct": 8.0},
    }
    queries = build_queries(legacy_stats)
    assert 2 <= len(queries) <= 4
    assert all(isinstance(q, str) and q for q in queries)


def test_build_queries_caps_at_max_queries():
    stats = _enriched_stats(
        {
            1: {"low_confidence": True},
            2: {"pct": 10.0, "low_confidence": True},
            3: {"pct": 2.0, "low_confidence": True},
            4: {"low_confidence": True},
        }
    )
    queries = build_queries(stats)
    assert len(queries) <= 4


class _FakeVectorstore:
    def __init__(self, responses: dict[str, list[Document]]):
        self._responses = responses
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.calls.append((query, k))
        return self._responses.get(query, [])


def test_retrieve_dedupes_across_queries():
    shared_doc = Document(page_content="Liver overview text.", metadata={"source": "liver-anatomy-overview.md"})
    other_doc = Document(page_content="Kidney overview text.", metadata={"source": "kidneys-anatomy-overview.md"})
    vectorstore = _FakeVectorstore(
        {
            "query one": [shared_doc, other_doc],
            "query two": [shared_doc],
        }
    )

    results = retrieve(["query one", "query two"], vectorstore, k=3)

    assert len(results) == 2
    assert results[0].metadata["source"] == "liver-anatomy-overview.md"
    assert results[1].metadata["source"] == "kidneys-anatomy-overview.md"


def test_retrieve_passes_k_through():
    vectorstore = _FakeVectorstore({"a query": []})
    retrieve(["a query"], vectorstore, k=5)
    assert vectorstore.calls == [("a query", 5)]
