"""
Read the test-set metrics JSONs written by evaluate.py and print a
README-ready markdown results table (SwinDAF vs. DAF baseline, per organ).

This script is the *only* source of truth for numbers in the README's
Results section — nothing there should ever be typed by hand.

Usage:
    python scripts/make_results_table.py
    python scripts/make_results_table.py --outputs_dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORGAN_ORDER = ["Liver", "Right Kidney", "Left Kidney", "Spleen"]
MODEL_LABELS = {"swin_daf": "SwinDAF (Swin-Tiny + DAF)", "daf": "DAF baseline (ResNeXt-101)"}


def load_latest_metrics(outputs_dir: Path) -> dict[str, dict]:
    """Return the most recent test_metrics_<model>_<timestamp>.json per model."""
    latest: dict[str, dict] = {}
    for path in sorted(outputs_dir.glob("test_metrics_*.json")):
        with open(path) as f:
            data = json.load(f)
        model = data.get("model")
        if model is None:
            continue
        # Files sort lexicographically by timestamp suffix, so the last one wins.
        latest[model] = data
    return latest


def format_table(metrics: dict[str, dict]) -> str:
    lines = []
    lines.append("### Test-set Dice score (2D per-slice, mean ± std)\n")
    header = "| Organ | " + " | ".join(MODEL_LABELS.get(m, m) for m in metrics) + " |"
    sep = "|---|" + "---|" * len(metrics)
    lines.append(header)
    lines.append(sep)
    for organ in ORGAN_ORDER:
        row = [organ]
        for model in metrics:
            v = metrics[model]["dice_2d_per_organ"].get(organ)
            row.append(f"{v['mean']:.3f} ± {v['std']:.3f}" if v else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    mean_row = ["**Mean**"]
    for model in metrics:
        mean_row.append(f"**{metrics[model]['mean_dice_2d']:.3f}**")
    lines.append("| " + " | ".join(mean_row) + " |")

    if all(metrics[m].get("dice_3d_per_organ") for m in metrics):
        lines.append("\n### Test-set Dice score (3D per-subject, mean ± std)\n")
        lines.append(header)
        lines.append(sep)
        for organ in ORGAN_ORDER:
            row = [organ]
            for model in metrics:
                v = metrics[model]["dice_3d_per_organ"].get(organ)
                row.append(f"{v['mean']:.3f} ± {v['std']:.3f}" if v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        mean_row = ["**Mean**"]
        for model in metrics:
            mean_row.append(f"**{metrics[model]['mean_dice_3d']:.3f}**")
        lines.append("| " + " | ".join(mean_row) + " |")
        n_subj = next(iter(metrics.values()))["num_subjects_3d"]
        lines.append(f"\n*3D scores computed per-subject over {n_subj} held-out test subjects.*")

    lines.append("\n### Provenance\n")
    for model in metrics:
        m = metrics[model]
        lines.append(
            f"- **{MODEL_LABELS.get(model, model)}**: checkpoint `{m['checkpoint']}`, "
            f"seed {m['seed']}, {m['num_slices']} test slices, evaluated {m['date_utc']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs_dir", default="outputs", type=str)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    metrics = load_latest_metrics(outputs_dir)
    if not metrics:
        raise SystemExit(f"No test_metrics_*.json files found in {outputs_dir}/ — run evaluate.py first.")

    table = format_table(metrics)
    print(table)

    out_path = outputs_dir / "results_table.md"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\n<!-- Wrote {out_path} -->")


if __name__ == "__main__":
    main()
