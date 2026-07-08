"""Unit tests for rag/eval_reports.py pure-logic functions - no GPU/network/model."""

from __future__ import annotations

from pathlib import Path

from rag.eval_reports import (
    any_organ_flagged,
    compute_auto_scores,
    has_expected_sections,
    mentions_uncertainty,
    select_slices,
    split_sentences,
    uses_reference,
)


def _enriched_stats(overrides: dict | None = None) -> dict:
    base = {
        "total": 224 * 224,
        0: {"pixels": 30000, "pct": 60.0, "low_confidence": False},
        1: {"pixels": 10000, "pct": 20.0, "low_confidence": False},
        2: {"pixels": 3000, "pct": 6.0, "low_confidence": False},
        3: {"pixels": 3000, "pct": 6.0, "low_confidence": False},
        4: {"pixels": 4000, "pct": 8.0, "low_confidence": False},
    }
    for cls_id, patch in (overrides or {}).items():
        base[cls_id] = {**base[cls_id], **patch}
    return base


def test_any_organ_flagged_true_when_one_organ_low_confidence():
    stats = _enriched_stats({2: {"low_confidence": True}})
    assert any_organ_flagged(stats) is True


def test_any_organ_flagged_false_when_none_flagged():
    stats = _enriched_stats()
    assert any_organ_flagged(stats) is False


def test_mentions_uncertainty_hits():
    assert mentions_uncertainty("This region requires human review.") is True
    assert mentions_uncertainty("Confidence is LOW - flag for review here.") is True


def test_mentions_uncertainty_misses():
    assert mentions_uncertainty("The liver appears normal with high confidence.") is False


def test_has_expected_sections_both_present():
    text = "Findings:\nLiver normal.\n\nImpression:\nNo acute findings."
    assert has_expected_sections(text) is True


def test_has_expected_sections_case_insensitive():
    text = "FINDINGS:\nLiver normal.\n\nIMPRESSION:\nNo acute findings."
    assert has_expected_sections(text) is True


def test_has_expected_sections_missing_one():
    text = "Findings:\nLiver normal.\n\nJust a summary, no label."
    assert has_expected_sections(text) is False


def test_has_expected_sections_missing_both():
    text = "The liver appears normal overall."
    assert has_expected_sections(text) is False


def test_uses_reference_true_when_references_block_present():
    text = "Findings:\n...\n\nImpression:\n...\n\n## References\n- liver-anatomy-overview.md"
    assert uses_reference(text) is True


def test_uses_reference_false_when_absent():
    text = "Findings:\n...\n\nImpression:\n..."
    assert uses_reference(text) is False


def test_split_sentences_excludes_references_block():
    text = (
        "Findings:\nThe liver is normal. The kidneys are symmetric.\n\n"
        "Impression:\nNo acute findings.\n\n"
        "## References\n- liver-anatomy-overview.md\n- kidneys-anatomy-overview.md"
    )
    sentences = split_sentences(text)
    joined = " ".join(sentences)
    assert "References" not in joined
    assert "liver-anatomy-overview" not in joined
    assert any("liver is normal" in s.lower() for s in sentences)


def test_split_sentences_empty_text_returns_empty_list():
    assert split_sentences("") == []
    assert split_sentences("## References\n- foo.md") == []


def test_select_slices_deterministic_same_seed():
    files = [Path(f"Subj_1slice_{i}.png") for i in range(1, 63)]
    first = select_slices(files, n=30, seed=42)
    second = select_slices(files, n=30, seed=42)
    assert first == second
    assert len(first) == 30


def test_select_slices_different_seed_can_differ():
    files = [Path(f"Subj_1slice_{i}.png") for i in range(1, 63)]
    a = select_slices(files, n=30, seed=42)
    b = select_slices(files, n=30, seed=43)
    assert a != b


def test_compute_auto_scores_zero_denominator_is_none_not_zero():
    reports = [
        {"any_organ_flagged": False, "uncertainty_mentioned": False, "structure_ok": True, "reference_usage_ok": None},
        {"any_organ_flagged": False, "uncertainty_mentioned": False, "structure_ok": True, "reference_usage_ok": None},
    ]
    scores = compute_auto_scores(reports, mode="norag")
    assert scores["uncertainty_flagging_rate"]["rate"] is None
    assert scores["uncertainty_flagging_rate"]["flagged_slices"] == 0
    assert "reference_usage_rate" not in scores


def test_compute_auto_scores_rag_mode_includes_reference_usage_rate():
    reports = [
        {"any_organ_flagged": True, "uncertainty_mentioned": True, "structure_ok": True, "reference_usage_ok": True},
        {"any_organ_flagged": False, "uncertainty_mentioned": False, "structure_ok": True, "reference_usage_ok": False},
    ]
    scores = compute_auto_scores(reports, mode="rag")
    assert scores["uncertainty_flagging_rate"]["rate"] == 1.0
    assert scores["structure_adherence_rate"] == 1.0
    assert scores["reference_usage_rate"] == 0.5
