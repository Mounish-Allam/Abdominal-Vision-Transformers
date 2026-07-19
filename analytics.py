"""SQLite logging for segmentation inferences (evaluate.py and src/demo.py).

Two tables:
  - inferences: one row per slice processed (model/encoder used, LLM provider
    if a report was generated, latency, timestamp).
  - organ_stats: one row per organ per inference (pixel coverage, confidence,
    entropy, low-confidence flag, and Dice when ground truth is available -
    only true during evaluate.py runs, never for a live user-uploaded slice).

Kept dependency-free (stdlib sqlite3 only) so it works identically in the
HF Space's CPU environment with no extra install.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "outputs" / "analytics.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inferences (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc   TEXT NOT NULL,
    slice_id        TEXT NOT NULL,
    source          TEXT NOT NULL,      -- 'evaluate' or 'demo'
    model_name      TEXT NOT NULL,      -- 'swin_daf' or 'daf'
    encoder_name    TEXT,
    llm_provider    TEXT,               -- NULL when no report was generated
    llm_model       TEXT,
    latency_seconds REAL
);

CREATE TABLE IF NOT EXISTS organ_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_id    INTEGER NOT NULL REFERENCES inferences(id),
    organ           TEXT NOT NULL,
    pixels          INTEGER,
    pct             REAL,
    mean_confidence REAL,
    mean_entropy    REAL,
    low_confidence  INTEGER,            -- 0/1
    dice            REAL                -- NULL unless ground truth was available
);

CREATE INDEX IF NOT EXISTS idx_organ_stats_inference ON organ_stats(inference_id);
CREATE INDEX IF NOT EXISTS idx_organ_stats_organ ON organ_stats(organ);
"""


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def log_inference(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    slice_id: str,
    source: str,
    model_name: str,
    encoder_name: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    latency_seconds: float | None = None,
    organ_stats: dict,
    dice_scores: dict | None = None,
) -> int:
    """Log one inference and its per-organ stats. Returns the new inference id.

    `organ_stats` is keyed by organ name (e.g. "Liver") with a dict containing
    at least "pixels" and "pct"; "mean_confidence"/"mean_entropy"/"low_confidence"
    are optional. `dice_scores`, if given, maps organ name -> Dice float.
    """
    init_db(db_path)
    dice_scores = dice_scores or {}

    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO inferences
                (timestamp_utc, slice_id, source, model_name, encoder_name,
                 llm_provider, llm_model, latency_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                slice_id,
                source,
                model_name,
                encoder_name,
                llm_provider,
                llm_model,
                latency_seconds,
            ),
        )
        inference_id = cur.lastrowid

        for organ, stats in organ_stats.items():
            conn.execute(
                """
                INSERT INTO organ_stats
                    (inference_id, organ, pixels, pct, mean_confidence,
                     mean_entropy, low_confidence, dice)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inference_id,
                    organ,
                    stats.get("pixels"),
                    stats.get("pct"),
                    stats.get("mean_confidence"),
                    stats.get("mean_entropy"),
                    (
                        int(stats["low_confidence"])
                        if "low_confidence" in stats and stats["low_confidence"] is not None
                        else None
                    ),
                    dice_scores.get(organ),
                ),
            )

    return inference_id
