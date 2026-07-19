"""
Retrieval layer for RAG-grounded clinical reports.

build_queries() is pure, framework-free Python: it turns per-organ
segmentation statistics into a short list of retrieval queries, deterministically,
so it is trivially unit-testable without a real FAISS index or embeddings.

retrieve() takes an already-loaded vectorstore (any object exposing
similarity_search(query, k)) so tests can pass a stub/mock instead of a real
FAISS index.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.build_index import EMBEDDING_MODEL_NAME, INDEX_DIR

ORGAN_NAMES = {1: "Liver", 2: "Right Kidney", 3: "Left Kidney", 4: "Spleen"}
PRESENCE_EPSILON_PCT = 0.5
ASYMMETRY_RATIO_THRESHOLD = 0.30
MAX_QUERIES = 4


class SimilaritySearchable(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]: ...


def load_vectorstore(index_dir: Path = INDEX_DIR) -> FAISS:
    """Load the FAISS index built by rag/build_index.py.

    allow_dangerous_deserialization=True is safe here: this index is built
    exclusively from our own committed knowledge_base/ files by
    rag/build_index.py - never from user-supplied or untrusted input.
    """
    # token=False: this is a public model and never needs auth. Without this, a stale/invalid
    # cached HF login token (e.g. from `hf auth login` for unrelated checkpoint uploads) can
    # get attached implicitly and cause a 401 on this request.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={"token": False}
    )
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )


def _is_low_confidence(organ_stats: dict) -> bool:
    if "low_confidence" in organ_stats:
        return bool(organ_stats["low_confidence"])
    return False


def build_queries(
    seg_stats: dict,
    low_confidence_threshold: float = 0.5,
    high_entropy_threshold: float = 1.0,
) -> list[str]:
    """Turn segmentation stats into 2-4 deterministic retrieval queries.

    Pure function, no LangChain/network dependency - safe to unit test in
    isolation. Tolerates the legacy compute_stats() shape (pixels/pct only,
    no mean_confidence/mean_entropy/low_confidence keys).
    """
    present_organs = [
        cls_id
        for cls_id in (1, 2, 3, 4)
        if seg_stats.get(cls_id, {}).get("pct", 0.0) > PRESENCE_EPSILON_PCT
    ]

    baseline_names = " ".join(ORGAN_NAMES[c] for c in present_organs) or "abdominal organ"
    baseline_query = f"normal {baseline_names} anatomy MRI"
    context_query = "radiology report structure abdominal MRI single slice coverage"

    flagged_organs = [c for c in present_organs if _is_low_confidence(seg_stats.get(c, {}))]

    size_queries = [
        f"{ORGAN_NAMES[c]} normal size range T2 MRI" for c in flagged_organs
    ]

    asymmetry_query = None
    if 2 in present_organs and 3 in present_organs:
        pct_right = seg_stats[2]["pct"]
        pct_left = seg_stats[3]["pct"]
        larger = max(pct_right, pct_left)
        if larger > 0:
            relative_diff = abs(pct_right - pct_left) / larger
            if relative_diff > ASYMMETRY_RATIO_THRESHOLD:
                asymmetry_query = "kidney left right asymmetry normal variation"

    uncertainty_query = None
    if flagged_organs:
        uncertainty_query = "AI segmentation confidence entropy uncertainty human review"

    # baseline + context are always included (guarantees >= 2 queries);
    # conditional queries fill remaining slots up to MAX_QUERIES, in priority
    # order: uncertainty > asymmetry > per-organ size.
    queries = [baseline_query, context_query]
    conditional = [q for q in (uncertainty_query, asymmetry_query) if q] + size_queries
    remaining_slots = MAX_QUERIES - len(queries)
    queries.extend(conditional[:remaining_slots])

    return queries


def retrieve(
    queries: list[str], vectorstore: SimilaritySearchable, k: int = 3
) -> list[Document]:
    """Run similarity_search per query, dedupe by (source, content), preserve order."""
    seen: set[tuple[str, str]] = set()
    results: list[Document] = []
    for query in queries:
        for doc in vectorstore.similarity_search(query, k=k):
            key = (doc.metadata.get("source", ""), doc.page_content)
            if key not in seen:
                seen.add(key)
                results.append(doc)
    return results
