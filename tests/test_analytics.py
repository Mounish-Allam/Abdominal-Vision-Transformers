"""Tests for analytics.py - real SQLite round-trips against a temp file, no GPU/network."""

from __future__ import annotations

import sqlite3

import analytics


def test_init_db_creates_expected_tables(tmp_path):
    db_path = tmp_path / "analytics.db"

    analytics.init_db(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"inferences", "organ_stats"} <= tables


def test_log_inference_writes_inference_row(tmp_path):
    db_path = tmp_path / "analytics.db"

    inference_id = analytics.log_inference(
        db_path,
        slice_id="Subj_1slice_1",
        source="demo",
        model_name="swin_daf",
        encoder_name="swin_tiny_patch4_window7_224",
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
        latency_seconds=0.18,
        organ_stats={"Liver": {"pixels": 1000, "pct": 20.0, "mean_confidence": 0.9}},
    )

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT slice_id, source, model_name, llm_provider FROM inferences WHERE id = ?",
            (inference_id,),
        ).fetchone()

    assert row == ("Subj_1slice_1", "demo", "swin_daf", "groq")


def test_log_inference_writes_one_organ_row_per_organ(tmp_path):
    db_path = tmp_path / "analytics.db"

    inference_id = analytics.log_inference(
        db_path,
        slice_id="Subj_1slice_1",
        source="evaluate",
        model_name="swin_daf",
        organ_stats={
            "Liver": {"pixels": 1000, "pct": 20.0, "mean_confidence": 0.9, "mean_entropy": 0.1},
            "Spleen": {"pixels": 200, "pct": 4.0, "mean_confidence": 0.5, "mean_entropy": 0.8},
        },
        dice_scores={"Liver": 0.95, "Spleen": 0.3},
    )

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT organ, dice, mean_confidence FROM organ_stats "
            "WHERE inference_id = ? ORDER BY organ",
            (inference_id,),
        ).fetchall()

    assert rows == [("Liver", 0.95, 0.9), ("Spleen", 0.3, 0.5)]


def test_dice_is_null_when_no_ground_truth_available(tmp_path):
    db_path = tmp_path / "analytics.db"

    inference_id = analytics.log_inference(
        db_path,
        slice_id="Subj_1slice_1",
        source="demo",
        model_name="swin_daf",
        organ_stats={"Liver": {"pixels": 1000, "pct": 20.0}},
    )

    with sqlite3.connect(str(db_path)) as conn:
        dice = conn.execute(
            "SELECT dice FROM organ_stats WHERE inference_id = ?", (inference_id,)
        ).fetchone()[0]

    assert dice is None


def test_low_confidence_flag_stored_as_integer(tmp_path):
    db_path = tmp_path / "analytics.db"

    inference_id = analytics.log_inference(
        db_path,
        slice_id="Subj_1slice_1",
        source="demo",
        model_name="swin_daf",
        organ_stats={"Liver": {"pixels": 1000, "pct": 20.0, "low_confidence": True}},
    )

    with sqlite3.connect(str(db_path)) as conn:
        flag = conn.execute(
            "SELECT low_confidence FROM organ_stats WHERE inference_id = ?", (inference_id,)
        ).fetchone()[0]

    assert flag == 1


def test_multiple_inferences_accumulate_across_calls(tmp_path):
    db_path = tmp_path / "analytics.db"

    for i in range(3):
        analytics.log_inference(
            db_path,
            slice_id=f"Subj_1slice_{i}",
            source="demo",
            model_name="swin_daf",
            organ_stats={"Liver": {"pixels": 1000, "pct": 20.0}},
        )

    with sqlite3.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM inferences").fetchone()[0]

    assert count == 3
