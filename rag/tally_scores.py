"""
Aggregate the manually-filled claim_scoring_sheet.csv with the auto-scores
already embedded in reports_norag_<ts>.json / reports_rag_<ts>.json into a
final before/after RAG evaluation table.

Refuses to run if any row in the sheet still has a blank verdict - this is
the mechanical guard against tallying a half-reviewed sheet. Never fabricates
a number: any zero-denominator metric renders as "N/A", never as 0%.

Usage:
    python rag/tally_scores.py
    python rag/tally_scores.py --outputs_dir outputs --sheet outputs/claim_scoring_sheet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VALID_VERDICTS = {"supported", "unsupported", "partially_supported"}


def load_latest_reports(outputs_dir: Path, mode: str) -> dict:
    matches = sorted(outputs_dir.glob(f"reports_{mode}_*.json"))
    if not matches:
        raise SystemExit(f"No reports_{mode}_*.json files found in {outputs_dir}/ -- run rag/eval_reports.py first.")
    with open(matches[-1]) as f:
        return json.load(f)


def load_scoring_sheet(sheet_path: Path) -> list[dict]:
    if not sheet_path.is_file():
        raise SystemExit(f"Scoring sheet not found: {sheet_path} -- run rag/eval_reports.py first.")
    with open(sheet_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    blank = [r for r in rows if not r.get("verdict", "").strip()]
    if blank:
        raise SystemExit(
            f"{len(blank)} of {len(rows)} rows in {sheet_path} have no verdict -- "
            "fill in the sheet before running tally_scores.py."
        )

    for row in rows:
        verdict = row["verdict"].strip().lower()
        if verdict not in VALID_VERDICTS:
            raise SystemExit(
                f"Row {row.get('row_id')} has unrecognized verdict '{row['verdict']}' -- "
                f"expected one of {sorted(VALID_VERDICTS)}."
            )
        row["verdict"] = verdict

    return rows


def unsupported_claim_rate(rows: list[dict], mode: str) -> dict:
    mode_rows = [r for r in rows if r["mode"] == mode]
    if not mode_rows:
        return {"rate": None, "unsupported": 0, "partially_supported": 0, "total": 0}
    unsupported = sum(1 for r in mode_rows if r["verdict"] == "unsupported")
    partial = sum(1 for r in mode_rows if r["verdict"] == "partially_supported")
    return {
        "rate": unsupported / len(mode_rows),
        "unsupported": unsupported,
        "partially_supported": partial,
        "total": len(mode_rows),
    }


def format_rate(rate: float | None, note: str = "N/A") -> str:
    return f"{rate * 100:.1f}%" if rate is not None else note


def format_table(norag: dict, rag: dict, claim_rows: list[dict]) -> tuple[str, list[str]]:
    regressions = []

    def delta_pp(before: float | None, after: float | None) -> str:
        if before is None or after is None:
            return "--"
        d = (after - before) * 100
        return f"{d:+.1f}pp"

    norag_unc = norag["auto_scores"]["uncertainty_flagging_rate"]
    rag_unc = rag["auto_scores"]["uncertainty_flagging_rate"]
    norag_struct = norag["auto_scores"]["structure_adherence_rate"]
    rag_struct = rag["auto_scores"]["structure_adherence_rate"]
    rag_refs = rag["auto_scores"].get("reference_usage_rate")

    norag_claims = unsupported_claim_rate(claim_rows, "norag")
    rag_claims = unsupported_claim_rate(claim_rows, "rag")

    if rag_struct < norag_struct:
        regressions.append(f"Structure adherence regressed: {norag_struct:.1%} -> {rag_struct:.1%}")
    if norag_unc["rate"] is not None and rag_unc["rate"] is not None and rag_unc["rate"] < norag_unc["rate"]:
        regressions.append(f"Uncertainty-flagging rate regressed: {norag_unc['rate']:.1%} -> {rag_unc['rate']:.1%}")
    if rag_claims["rate"] is not None and norag_claims["rate"] is not None and rag_claims["rate"] > norag_claims["rate"]:
        regressions.append(f"Unsupported-claim rate regressed: {norag_claims['rate']:.1%} -> {rag_claims['rate']:.1%}")

    lines = [
        f"### Report grounding evaluation (before/after RAG, n={norag['n']} slices, seed {norag['seed']})\n",
        "| Metric | No-RAG (legacy) | RAG-grounded | Delta |",
        "|---|---|---|---|",
        f"| Structure adherence (Findings:/Impression: present) | {norag_struct:.1%} | {rag_struct:.1%} | {delta_pp(norag_struct, rag_struct)} |",
        f"| Uncertainty-flagging rate (when any organ flagged) | {format_rate(norag_unc['rate'])} | {format_rate(rag_unc['rate'])} | {delta_pp(norag_unc['rate'], rag_unc['rate'])} |",
        f"| Reference usage (>= 1 passage cited) | N/A | {format_rate(rag_refs)} | -- |",
        f"| Unsupported-claim rate (manual review) | {format_rate(norag_claims['rate'])} ({norag_claims['unsupported']}/{norag_claims['total']}) | {format_rate(rag_claims['rate'])} ({rag_claims['unsupported']}/{rag_claims['total']}) | {delta_pp(norag_claims['rate'], rag_claims['rate'])} |",
        "",
        f"*Unsupported-claim rate is the fraction of sentences marked `unsupported` in "
        f"`outputs/claim_scoring_sheet.csv`, out of {norag_claims['total']} no-RAG / "
        f"{rag_claims['total']} RAG sentences. See the scoring methodology note for how "
        f"this sheet was filled in. "
        f"Partially-supported sentences ({norag_claims['partially_supported']} no-RAG / "
        f"{rag_claims['partially_supported']} RAG) are tracked separately, not counted "
        f"as unsupported.*",
    ]

    if regressions:
        lines.append("\n**Regressions:**")
        for r in regressions:
            lines.append(f"- REGRESSION: {r}")

    return "\n".join(lines), regressions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs_dir", default="outputs", type=str)
    parser.add_argument("--sheet", default="outputs/claim_scoring_sheet.csv", type=str)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    norag = load_latest_reports(outputs_dir, "norag")
    rag = load_latest_reports(outputs_dir, "rag")
    claim_rows = load_scoring_sheet(Path(args.sheet))

    table, regressions = format_table(norag, rag, claim_rows)
    print(table)
    if regressions:
        print(f"\n{len(regressions)} regression(s) detected - see above.")

    out_path = outputs_dir / "rag_eval_table.md"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\n<!-- Wrote {out_path} -->")


if __name__ == "__main__":
    main()
