"""
Generate before/after RAG clinical reports for a fixed, seeded set of test slices.

For --n test slices (seeded selection, deterministic across runs), runs the
trained checkpoint to get segmentation stats, then generates a report in both
modes (legacy/no-RAG and RAG-grounded) via report_generator.generate_report().
Writes two matched JSON files (one per mode) with per-slice results and
auto-scored metrics, plus a claim_scoring_sheet.csv for manual review - the
claim-scoring verdict columns are always left blank; a language model grading
its own or a sibling model's claims would reintroduce the same fabrication
risk this evaluation exists to measure.

Usage:
    python rag/eval_reports.py --n 30 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from demo import load_model, preprocess, run_model, compute_stats  # noqa: E402
from report_generator import generate_report  # noqa: E402

FILENAME_RE = re.compile(r"Subj_(\d+)slice_(\d+)\.png")

UNCERTAINTY_KEYWORDS = [
    "human review",
    "requires review",
    "require review",
    "flag for review",
    "further review",
    "clinical correlation",
    "uncertain",
    "uncertainty",
    "low confidence",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_slices(file_list: list[Path], n: int, seed: int) -> list[Path]:
    """Deterministically select n files from a sorted file_list given a seed."""
    rng = random.Random(seed)
    return rng.sample(sorted(file_list), n)


def parse_subject_slice(path: Path) -> tuple[str, int] | None:
    match = FILENAME_RE.search(path.name)
    if not match:
        return None
    subj_id, slice_num = match.groups()
    return subj_id, int(slice_num)


def any_organ_flagged(organ_stats: dict) -> bool:
    return any(organ_stats.get(c, {}).get("low_confidence", False) for c in (1, 2, 3, 4))


def mentions_uncertainty(report_text: str) -> bool:
    text = report_text.lower()
    return any(keyword in text for keyword in UNCERTAINTY_KEYWORDS)


def has_expected_sections(report_text: str) -> bool:
    return bool(
        re.search(r"findings\s*:", report_text, re.IGNORECASE)
        and re.search(r"impression\s*:", report_text, re.IGNORECASE)
    )


def uses_reference(report_text: str) -> bool:
    return "## References" in report_text


def strip_references_block(report_text: str) -> str:
    return report_text.split("## References")[0]


def split_sentences(report_text: str) -> list[str]:
    """Simple sentence splitter - no NLP dependency. Known limitation: doesn't
    handle abbreviations (e.g. "Fig. 1") specially; acceptable for a manually
    reviewed sheet of a few hundred rows."""
    body = strip_references_block(report_text).strip()
    if not body:
        return []
    sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
    return [s.strip() for s in sentence_re.split(body) if s.strip()]


def compute_auto_scores(reports: list[dict], mode: str) -> dict:
    flagged = [r for r in reports if r["any_organ_flagged"]]
    mentioned_when_flagged = [r for r in flagged if r["uncertainty_mentioned"]]

    if flagged:
        uncertainty_rate = {
            "flagged_slices": len(flagged),
            "mentioned_when_flagged": len(mentioned_when_flagged),
            "rate": len(mentioned_when_flagged) / len(flagged),
            "note": None,
        }
    else:
        uncertainty_rate = {
            "flagged_slices": 0,
            "mentioned_when_flagged": 0,
            "rate": None,
            "note": "no organ exceeded the confidence/entropy threshold in any selected slice",
        }

    scores = {
        "uncertainty_flagging_rate": uncertainty_rate,
        "structure_adherence_rate": sum(r["structure_ok"] for r in reports) / len(reports),
    }

    if mode == "rag":
        scores["reference_usage_rate"] = sum(
            bool(r["reference_usage_ok"]) for r in reports
        ) / len(reports)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", default=30, type=int, help="Number of test slices to sample")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weights", default="model/Best_SwinDAF-CHAOS.pth", type=str)
    parser.add_argument(
        "--encoder",
        default="swin_tiny_patch4_window7_224",
        choices=[
            "swin_tiny_patch4_window7_224",
            "swin_small_patch4_window7_224",
            "swin_base_patch4_window7_224",
        ],
        type=str,
    )
    parser.add_argument("--test_dir", default="DataSet/test/Img", type=str)
    parser.add_argument("--api_key", default="", type=str, help="Falls back to GROQ_API_KEY env var")
    parser.add_argument("--out_dir", default="outputs", type=str)
    parser.add_argument("--sleep", default=1.0, type=float, help="Seconds between Groq calls")
    args = parser.parse_args()

    set_seed(args.seed)

    file_list = sorted((REPO_ROOT / args.test_dir).glob("*.png"))
    if len(file_list) < args.n:
        raise SystemExit(f"Only {len(file_list)} files found in {args.test_dir}, need --n={args.n}")
    selected = select_slices(file_list, args.n, args.seed)

    subject_counts: dict[str, int] = {}
    slice_ids = []
    for path in selected:
        parsed = parse_subject_slice(path)
        if parsed is None:
            raise SystemExit(f"Filename does not match Subj_<id>slice_<n>.png: {path.name}")
        subj_id, slice_num = parsed
        subject_counts[f"Subj_{subj_id}"] = subject_counts.get(f"Subj_{subj_id}", 0) + 1
        slice_ids.append(f"Subj_{subj_id}slice_{slice_num}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {args.weights} ({args.encoder})")
    model = load_model(args.weights, args.encoder, device)

    results = {"norag": [], "rag": []}
    csv_rows = []
    row_id = 0

    for i, (path, slice_id) in enumerate(zip(selected, slice_ids)):
        print(f"[{i + 1}/{len(selected)}] {slice_id}")
        img = Image.open(path)
        tensor = preprocess(img)
        mask, probs = run_model(model, tensor, device)
        organ_stats = compute_stats(mask, probs)
        flagged = any_organ_flagged(organ_stats)

        for mode in ("norag", "rag"):
            report_md, passages_md = generate_report(
                organ_stats, api_key=args.api_key, use_rag=(mode == "rag")
            )
            time.sleep(args.sleep)

            entry = {
                "slice_id": slice_id,
                "image_path": str(path.relative_to(REPO_ROOT)),
                "organ_stats": organ_stats,
                "report_markdown": report_md,
                "passages_markdown": passages_md,
                "any_organ_flagged": flagged,
                "uncertainty_mentioned": mentions_uncertainty(report_md),
                "structure_ok": has_expected_sections(report_md),
                "reference_usage_ok": uses_reference(report_md) if mode == "rag" else None,
            }
            results[mode].append(entry)

            for sentence in split_sentences(report_md):
                row_id += 1
                csv_rows.append(
                    {
                        "row_id": row_id,
                        "slice_id": slice_id,
                        "mode": mode,
                        "sentence": sentence,
                        "supported_by_measurement?": "",
                        "supported_by_passage?": "",
                        "verdict": "",
                    }
                )

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for mode in ("norag", "rag"):
        payload = {
            "mode": mode,
            "seed": args.seed,
            "n": args.n,
            "weights": args.weights,
            "encoder": args.encoder,
            "date_utc": timestamp,
            "selected_slice_ids": sorted(slice_ids),
            "subject_counts": subject_counts,
            "reports": results[mode],
            "auto_scores": compute_auto_scores(results[mode], mode),
        }
        out_path = out_dir / f"reports_{mode}_{timestamp}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out_path}")

    csv_path = out_dir / "claim_scoring_sheet.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id", "slice_id", "mode", "sentence",
                "supported_by_measurement?", "supported_by_passage?", "verdict",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path} ({len(csv_rows)} rows - fill in supported_by_measurement?/supported_by_passage?/verdict)")


if __name__ == "__main__":
    main()
