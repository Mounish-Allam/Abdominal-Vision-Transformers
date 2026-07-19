from __future__ import annotations

import sys
from pathlib import Path

try:
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.provider import ProviderError, get_llm_client  # noqa: E402

ORGAN_NAMES = {1: "Liver", 2: "Right Kidney", 3: "Left Kidney", 4: "Spleen"}

CONFIDENCE_THRESHOLD: float = 0.5
ENTROPY_THRESHOLD: float = 1.0  # ln(5) ~= 1.609 is max entropy for 5 classes;
                                 # 1.0 is ~62% of max - flags meaningfully uncertain regions.

_GROUNDED_SYSTEM = (
    "You are a radiologist assistant. Using ONLY the reference passages and the "
    "measurements provided below, write a concise clinical impression (under 200 "
    "words) about a single 2D abdominal MRI slice's automated segmentation output. "
    "Use formal radiology language. Structure your response as exactly two labeled "
    "sections: 'Findings:' followed by objective per-organ observations, and "
    "'Impression:' followed by your overall clinical impression. If an organ's "
    "confidence is below the stated threshold or its entropy is high, explicitly "
    "state that region requires human review. Do not state findings that are not "
    "supported by the reference passages or the measurements - do not speculate "
    "beyond what is given."
)

_LEGACY_SYSTEM = (
    "You are a radiologist assistant. Based on 2D abdominal MRI segmentation results, "
    "write a concise clinical impression (under 180 words). Use formal radiology language. "
    "Structure your response as exactly two labeled sections: 'Findings:' followed by "
    "objective observations, and 'Impression:' followed by your overall clinical "
    "impression. Comment on organ presence, relative coverage compared to typical "
    "anatomy, and any notable asymmetries between paired structures."
)

_vectorstore_cache = None


def _format_measurements(organ_stats: dict) -> str:
    total = organ_stats.get("total", 1)
    bg_pct = organ_stats.get(0, {}).get("pct", 0.0)

    lines = []
    for cls_id, stats in organ_stats.items():
        if not isinstance(cls_id, int) or cls_id == 0:
            continue
        line = f"- {ORGAN_NAMES[cls_id]}: {stats['pixels']:,} px ({stats['pct']:.1f}% of slice)"
        if "mean_confidence" in stats and stats["mean_confidence"] is not None:
            conf = stats["mean_confidence"]
            ent = stats["mean_entropy"]
            low_conf = stats.get("low_confidence", False)
            line += f", mean confidence {conf:.2f}"
            if low_conf:
                line += " (LOW - flag for human review)"
            line += f", mean entropy {ent:.2f}"
            if ent > ENTROPY_THRESHOLD:
                line += " (HIGH)"
        lines.append(line)

    return (
        "\n".join(lines)
        + f"\n- Background: {bg_pct:.1f}% of slice"
        + f"\n- Total pixels: {total:,}"
    )


def _get_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is None:
        from rag.retrieval import load_vectorstore

        _vectorstore_cache = load_vectorstore()
    return _vectorstore_cache


def _format_passages_block(passages: list) -> str:
    blocks = []
    for doc in passages:
        source = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "")
        blocks.append(f"[{source}] ({topic})\n{doc.page_content}")
    return "\n\n".join(blocks)


def _format_passages_markdown(passages: list) -> str:
    if not passages:
        return ""
    blocks = []
    for doc in passages:
        source = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "")
        blocks.append(f"**{source}** ({topic})\n\n{doc.page_content}\n")
    return "\n---\n".join(blocks)


def _format_references(passages: list) -> str:
    if not passages:
        return ""
    seen = []
    for doc in passages:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.append(source)
    return "\n\n## References\n" + "\n".join(f"- {name}" for name in seen)


def _grounded_prompt(measurements: str, passages: list) -> ChatPromptTemplate:
    passages_block = _format_passages_block(passages) if passages else "(no reference passages retrieved)"
    return ChatPromptTemplate.from_messages(
        [
            ("system", _GROUNDED_SYSTEM),
            (
                "human",
                "Reference passages:\n{passages_block}\n\n"
                "Segmentation measurements:\n{measurements}\n\n"
                "Generate the clinical impression now.",
            ),
        ]
    ).partial(passages_block=passages_block, measurements=measurements)


def _legacy_prompt(measurements: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _LEGACY_SYSTEM),
            (
                "human",
                "Single 2D abdominal MRI slice - automated segmentation output:\n\n"
                "{measurements}\n\n"
                "Generate a brief radiologist-style clinical impression.",
            ),
        ]
    ).partial(measurements=measurements)


def generate_report(
    organ_stats: dict, api_key: str = "", use_rag: bool = True
) -> tuple[str, str]:
    """Generate a clinical report from segmentation stats.

    Returns (report_markdown, passages_markdown). passages_markdown is "" when
    use_rag is False or no passages were retrieved/available.
    """
    if not LANGCHAIN_AVAILABLE:
        return "Run: pip install langchain-groq langchain-openai langchain-core", ""

    try:
        llm = get_llm_client(api_key=api_key)
    except ProviderError as exc:
        return str(exc), ""

    measurements = _format_measurements(organ_stats)
    passages_md = ""

    try:
        passages: list = []
        if use_rag:
            try:
                from rag.retrieval import build_queries, retrieve

                vectorstore = _get_vectorstore()
                queries = build_queries(organ_stats)
                passages = retrieve(queries, vectorstore, k=3)
                prompt = _grounded_prompt(measurements, passages)
                passages_md = _format_passages_markdown(passages)
            except FileNotFoundError:
                prompt = _legacy_prompt(measurements)
                passages_md = (
                    "_RAG unavailable: index not found - run `python rag/build_index.py`. "
                    "Showing ungrounded report._"
                )
        else:
            prompt = _legacy_prompt(measurements)

        messages = prompt.format_messages()
        response = llm.invoke(messages)
        text = response.content

        if use_rag and passages:
            text += _format_references(passages)

        return text, passages_md
    except Exception as exc:
        return f"Report generation failed: {exc}", ""
