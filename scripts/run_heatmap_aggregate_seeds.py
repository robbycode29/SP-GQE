#!/usr/bin/env python3
"""Replay every seed in ``AGGREGATED_SUMMARY.json`` with heatmaps ON.

Each invocation matches one aggregate row: ``--seed`` + ``--sample-size``. Omit
``--no-heatmap`` so ``run_experiment.py`` builds the n×τ grids (single-seed rule).

Across all rows this reproduces the **562** stratified instances used in the
aggregate (23 seeds). Run across multiple UTC days if Groq **TPD** requires.

After each successful experiment, prints **heatmap coverage** toward 562: only
rows that have a ``heatmap`` block in ``daily_runs`` (latest ``date_utc`` per
seed) count.

When a session finishes all seeds successfully, runs **merge heatmaps** +
**aggregate_daily_runs** unless disabled.

Examples::

    python scripts/run_heatmap_aggregate_seeds.py
    python scripts/run_heatmap_aggregate_seeds.py --skip-until 10
    python scripts/run_heatmap_aggregate_seeds.py --dry-run
    python scripts/run_heatmap_aggregate_seeds.py --finalize-only
    python scripts/run_heatmap_aggregate_seeds.py --report-after-each-seed

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _heatmap_coverage_for_summary_rows(
    target_rows: list[dict],
    daily_dir: Path,
) -> tuple[int, int]:
    """(covered_question_instances, target) for summary rows whose *latest* JSON has a heatmap."""
    target = sum(int(r["sample_size"]) for r in target_rows)
    groups: dict[tuple[int, int], list[tuple[str, dict]]] = defaultdict(list)
    for path in sorted(daily_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sk = int(data.get("seed", -1))
        n = int(data.get("sample_size", 0))
        groups[(sk, n)].append((str(data.get("date_utc", "")), data))
    keys_with_hm: set[tuple[int, int]] = set()
    for key, items in groups.items():
        items.sort(key=lambda x: x[0])
        latest = items[-1][1]
        if isinstance(latest.get("heatmap"), dict):
            keys_with_hm.add(key)
    covered = sum(
        int(r["sample_size"])
        for r in target_rows
        if (int(r["seed"]), int(r["sample_size"])) in keys_with_hm
    )
    return covered, target


def _print_coverage(label: str, rows: list[dict], daily_dir: Path) -> None:
    cov, tgt = _heatmap_coverage_for_summary_rows(rows, daily_dir)
    print(f"{label} Heatmap coverage (target rows): {cov}/{tgt}", flush=True)


def _finalize(*, daily_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "merge_heatmaps.py"),
            "--daily-dir",
            str(daily_dir),
        ],
        cwd=REPO,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "aggregate_daily_runs.py"),
            "--daily-dir",
            str(daily_dir),
        ],
        cwd=REPO,
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        type=Path,
        default=REPO / "results" / "AGGREGATED_SUMMARY.json",
        help="Aggregate JSON with seeds_used[] (defines target 562 rows)",
    )
    ap.add_argument(
        "--daily-dir",
        type=Path,
        default=REPO / "results" / "daily_runs",
        help="Where run_experiment writes per-seed JSON",
    )
    ap.add_argument(
        "--skip-until",
        type=int,
        default=1,
        metavar="N",
        help="1-based index into seeds_used to start from (after failures)",
    )
    ap.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only run merge_heatmaps.py + aggregate_daily_runs.py (no experiments)",
    )
    ap.add_argument(
        "--no-auto-finalize",
        action="store_true",
        help="After the last seed in this session, do not merge/aggregate",
    )
    ap.add_argument(
        "--report-after-each-seed",
        action="store_true",
        help="After each successful seed run, merge heatmaps + aggregate (slow; good end-of-day)",
    )
    ap.add_argument(
        "--merge-after",
        action="store_true",
        help="Deprecated alias: same as successful completion finalize; prefer default auto-finalize",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    rows: list[dict] = data["seeds_used"]
    args.daily_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        if args.dry_run:
            print("Would run merge_heatmaps.py then aggregate_daily_runs.py")
            return
        _finalize(daily_dir=args.daily_dir)
        _print_coverage("After finalize:", rows, args.daily_dir)
        return

    run_py = REPO / "scripts" / "run_experiment.py"
    _print_coverage("Before session:", rows, args.daily_dir)

    last_completed_index = 0
    any_run = False
    for i, row in enumerate(rows, start=1):
        if i < args.skip_until:
            continue
        seed = int(row["seed"])
        n = int(row["sample_size"])
        cmd = [
            sys.executable,
            str(run_py),
            "--stack",
            "plan",
            "--seed",
            str(seed),
            "--sample-size",
            str(n),
        ]
        print(
            f"[seed row {i}/{len(rows)}] heatmap run: seed={seed} sample_size={n}",
            flush=True,
        )
        if args.dry_run:
            print(" ", " ".join(cmd))
            continue
        any_run = True
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(
                f"Stopped at index {i}. Resume with: --skip-until {i}",
                file=sys.stderr,
            )
            _print_coverage("Stopped with coverage:", rows, args.daily_dir)
            sys.exit(r.returncode)
        last_completed_index = i
        _print_coverage("Progress:", rows, args.daily_dir)
        if args.report_after_each_seed:
            _finalize(daily_dir=args.daily_dir)

    if args.dry_run:
        return

    completed_all = last_completed_index == len(rows)
    do_finalize = (
        any_run or args.merge_after
    ) and completed_all and not args.no_auto_finalize
    if do_finalize:
        _finalize(daily_dir=args.daily_dir)
        _print_coverage("After finalize:", rows, args.daily_dir)


if __name__ == "__main__":
    main()
