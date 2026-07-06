"""
Build the FAISS vector index for RAG-grounded clinical reports.

Loads knowledge_base/*.md (each with a YAML front-matter header of
source_url/license/topic), splits into overlapping chunks, embeds them with a
local sentence-transformers model, and persists a FAISS index to rag/kb_index/.

CPU-only, no network calls besides the (cached, one-time) embedding model
download from Hugging Face. Runs in well under 2 minutes on this project's
knowledge_base/ size.

Run:
    python rag/build_index.py
    python rag/build_index.py --kb_dir knowledge_base --out_dir rag/kb_index
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "knowledge_base"
INDEX_DIR = REPO_ROOT / "rag" / "kb_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a leading '---\\n...\\n---\\n' YAML block into a flat dict + body.

    Only handles flat scalar keys (source_url, license, topic) - the
    knowledge_base/ front matter never nests, so a full YAML parser is not
    needed and avoids adding a PyYAML dependency for three string fields.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    lines = stripped.splitlines()
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text

    metadata: dict = {}
    for line in lines[1:end_idx]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip().strip('"').strip("'")

    body = "\n".join(lines[end_idx + 1 :])
    return metadata, body.strip()


def load_documents(kb_dir: Path = KB_DIR) -> list[Document]:
    """Load knowledge_base/*.md, stripping front matter into metadata."""
    if not kb_dir.is_dir() or not any(kb_dir.glob("*.md")):
        raise FileNotFoundError(
            f"No markdown files found in {kb_dir} - populate knowledge_base/ first."
        )

    loader = DirectoryLoader(
        str(kb_dir),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()

    documents: list[Document] = []
    for raw in raw_docs:
        front_matter, body = _parse_front_matter(raw.page_content)
        source_name = Path(raw.metadata.get("source", "")).name
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "source": source_name,
                    "source_url": front_matter.get("source_url", "N/A"),
                    "license": front_matter.get("license", "N/A"),
                    "topic": front_matter.get("topic", ""),
                },
            )
        )
    return documents


def split_documents(docs: list[Document]) -> list[Document]:
    """Chunk documents on sentence-friendly boundaries, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(kb_dir: Path = KB_DIR, out_dir: Path = INDEX_DIR) -> None:
    """Load, split, embed, and persist the FAISS index."""
    docs = load_documents(kb_dir)
    chunks = split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_dir))

    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks.")
    print(f"Index saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kb_dir", default=str(KB_DIR), type=str)
    parser.add_argument("--out_dir", default=str(INDEX_DIR), type=str)
    args = parser.parse_args()

    start = time.time()
    build_index(Path(args.kb_dir), Path(args.out_dir))
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
