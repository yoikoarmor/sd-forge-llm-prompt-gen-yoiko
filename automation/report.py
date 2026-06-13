"""Quality report over accumulated runs.

Usage:
    python report.py
    python report.py --csv export.csv
    python report.py --worst 20
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

AUTOMATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTOMATION_DIR))

import db


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(v, digits=2):
    return f"{v:.{digits}f}" if v is not None else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--worst", type=int, default=10)
    args = parser.parse_args()

    with open(AUTOMATION_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    conn = db.connect_db(
        AUTOMATION_DIR / config["paths"]["output_dir"] / "runs.sqlite3"
    )
    rows = conn.execute(
        "SELECT id, created_at, gen_prompt, parts_json, llm_mode, "
        "clip_score, vlm_overall, vlm_json, disc_score, image_path "
        "FROM runs ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        print("No runs recorded yet.")
        return

    total = len(rows)
    fallback = sum(1 for r in rows if r[4] == "llm fallback")
    clip_scores = [r[5] for r in rows]
    vlm_scores = [r[6] for r in rows]

    print("=" * 60)
    print(f"Total runs:        {total}")
    print(f"LLM fallback rate: {fallback}/{total} ({100 * fallback / total:.1f}%)")
    print(f"Mean CLIP score:   {_fmt(_mean(clip_scores), 4)}")
    print(f"Mean VLM overall:  {_fmt(_mean(vlm_scores))}")
    print("=" * 60)

    # Per-axis VLM means
    axis_values = defaultdict(list)
    for r in rows:
        if r[7]:
            try:
                scores = json.loads(r[7])
            except json.JSONDecodeError:
                continue
            for axis in ("prompt_fidelity", "anatomy", "artifacts", "aesthetics"):
                if isinstance(scores.get(axis), (int, float)):
                    axis_values[axis].append(scores[axis])
    if axis_values:
        print("\nVLM per-axis means:")
        for axis, values in axis_values.items():
            print(f"  {axis:16s} {_fmt(_mean(values))}")

    # Weak spots: mean vlm_overall per wordpool item (min 3 samples)
    category_scores = defaultdict(list)
    for r in rows:
        if r[3] and r[6] is not None:
            try:
                parts = json.loads(r[3])
            except json.JSONDecodeError:
                continue
            for category, value in parts.items():
                category_scores[(category, value)].append(r[6])
    weak = [
        (cat, val, _mean(scores), len(scores))
        for (cat, val), scores in category_scores.items()
        if len(scores) >= 3
    ]
    weak.sort(key=lambda x: x[2])
    if weak:
        print("\nWeakest wordpool items (mean VLM overall, n >= 3):")
        for cat, val, mean, n in weak[:10]:
            print(f"  {mean:.2f} (n={n}) [{cat}] {val}")

    # Worst individual runs
    scored = [r for r in rows if r[6] is not None]
    scored.sort(key=lambda r: r[6])
    if scored:
        print(f"\nWorst {min(args.worst, len(scored))} runs by VLM overall:")
        for r in scored[: args.worst]:
            print(f"  run {r[0]}: overall={r[6]:.0f} clip={_fmt(r[5], 3)}")
            print(f"    gen_prompt: {r[2]}")
            print(f"    image: {Path(r[9]).name if r[9] else '-'}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id", "created_at", "gen_prompt", "llm_mode",
                    "clip_score", "vlm_overall", "disc_score", "image_path",
                ]
            )
            for r in rows:
                writer.writerow([r[0], r[1], r[2], r[4], r[5], r[6], r[8], r[9]])
        print(f"\nExported {total} rows to {args.csv}")


if __name__ == "__main__":
    main()
