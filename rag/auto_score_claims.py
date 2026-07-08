"""
Automated, rule-based cross-check of claim_scoring_sheet.csv against the real
organ_stats measurements and retrieved passages.

This is NOT independent human clinical review. It is a deterministic,
auditable heuristic: extract numeric claims (percentages, confidence,
entropy) from each sentence via regex, and check whether they match a real
measurement within tolerance. It cannot judge clinical reasoning quality,
only whether a stated number is real or invented. Any project write-up using
this output must disclose it was scored this way, not by a human reviewer.

Usage:
    python rag/auto_score_claims.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?")
PCT_TOLERANCE = 0.5       # percentage points
VALUE_TOLERANCE = 0.05    # confidence/entropy units

ORGAN_NAME_TO_CLASS = {
    "liver": 1,
    "right kidney": 2,
    "left kidney": 3,
    "spleen": 4,
}


def load_latest_reports(outputs_dir: Path, mode: str) -> dict:
    matches = sorted(outputs_dir.glob(f"reports_{mode}_*.json"))
    if not matches:
        raise SystemExit(f"No reports_{mode}_*.json files found in {outputs_dir}/")
    with open(matches[-1]) as f:
        return json.load(f)


def build_report_index(norag: dict, rag: dict) -> dict:
    """Map (slice_id, mode) -> report entry, for sentence-level lookup."""
    index = {}
    for mode, data in (("norag", norag), ("rag", rag)):
        for report in data["reports"]:
            index[(report["slice_id"], mode)] = report
    return index


def _all_real_numbers(organ_stats: dict) -> list[float]:
    """Every numeric measurement present in organ_stats, for fuzzy matching."""
    values = []
    for key, stats in organ_stats.items():
        if key == "total" or not isinstance(stats, dict):
            continue
        for field in ("pct", "mean_confidence", "mean_entropy"):
            v = stats.get(field)
            if v is not None:
                values.append(float(v))
    return values


def number_is_real(number: float, organ_stats: dict) -> bool:
    real_numbers = _all_real_numbers(organ_stats)
    for real in real_numbers:
        if abs(number - real) <= PCT_TOLERANCE:
            return True
        if abs(number - real * 100) <= PCT_TOLERANCE:  # fraction vs percent mismatch
            return True
        if abs(number / 100 - real) <= VALUE_TOLERANCE:
            return True
        if abs(number - real) <= VALUE_TOLERANCE:
            return True
    return False


def score_measurement_support(sentence: str, organ_stats: dict) -> str:
    """yes / no / n/a - does every number mentioned match a real measurement?"""
    numbers = [float(n) for n in NUMBER_RE.findall(sentence)]
    if not numbers:
        return "n/a"
    results = [number_is_real(n, organ_stats) for n in numbers]
    if all(results):
        return "yes"
    if any(results):
        return "partial"
    return "no"


def score_passage_support(sentence: str, passages_markdown: str) -> str:
    """yes / no / n/a - does a retrieved passage share substantial wording with this sentence?"""
    if not passages_markdown:
        return "n/a"
    sentence_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", sentence)}
    passage_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", passages_markdown)}
    if not sentence_words:
        return "n/a"
    overlap = sentence_words & passage_words
    return "yes" if len(overlap) / len(sentence_words) >= 0.35 else "no"


def derive_verdict(measurement_support: str, passage_support: str, mode: str) -> str:
    if measurement_support == "no":
        return "unsupported"
    if measurement_support == "partial":
        return "partially_supported"
    if mode == "rag" and passage_support == "no" and measurement_support == "n/a":
        return "partially_supported"
    return "supported"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs_dir", default="outputs", type=str)
    parser.add_argument("--sheet", default="outputs/claim_scoring_sheet.csv", type=str)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    norag = load_latest_reports(outputs_dir, "norag")
    rag = load_latest_reports(outputs_dir, "rag")
    report_index = build_report_index(norag, rag)

    sheet_path = Path(args.sheet)
    with open(sheet_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        report = report_index.get((row["slice_id"], row["mode"]))
        if report is None:
            continue
        measurement_support = score_measurement_support(row["sentence"], report["organ_stats"])
        passage_support = score_passage_support(row["sentence"], report.get("passages_markdown", ""))
        verdict = derive_verdict(measurement_support, passage_support, row["mode"])

        row["supported_by_measurement?"] = measurement_support
        row["supported_by_passage?"] = passage_support
        row["verdict"] = verdict

    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
    print(f"Scored {len(rows)} rows -> {sheet_path}")
    print("Verdict counts:", verdict_counts)


if __name__ == "__main__":
    main()
