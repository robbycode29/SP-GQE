#!/usr/bin/env python3
"""Merge SP-GQE n×τ heatmaps from multiple per-seed `daily_runs/*.json` files.

Each contributing file must contain a top-level ``heatmap`` block with
``mean_f1_grid``, ``mean_retrieval_p_at_k_grid``, and matching ``n_hops`` /
``tau`` arrays (as produced by ``run_experiment.py`` with heatmaps enabled).

Grids are combined with a **sample-size-weighted** average (each file's cells
are already per-seed means over that seed's questions).

Outputs:
  ``results/merged_heatmap/merged_heatmap.json``
  ``results/merged_heatmap/heatmap_merged_mean_f1.png``
  ``results/merged_heatmap/heatmap_merged_mean_p_at_k.png``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sp_gqe.experiment.plots import fungi_heatmap  # noqa: E402


def _load_runs(daily_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(daily_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data.get("heatmap"), dict):
            continue
        out.append((path, data))
    return out


def _dedupe_latest_date(
    runs: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, dict[str, Any]]]:
    """One file per (seed, sample_size): keep the latest ``date_utc`` (re-runs)."""
    groups: dict[tuple[int, int], list[tuple[Path, dict[str, Any]]]] = defaultdict(
        list
    )
    for path, data in runs:
        sk = int(data.get("seed", -1))
        n = int(data.get("sample_size", 0))
        groups[(sk, n)].append((path, data))
    picked: list[tuple[Path, dict[str, Any]]] = []
    for key in sorted(groups):
        items = groups[key]
        items.sort(key=lambda x: str(x[1].get("date_utc", "")))
        picked.append(items[-1])
    return picked


def heatmap_coverage_for_summary_rows(
    target_rows: list[dict[str, Any]],
    daily_dir: Path,
) -> tuple[int, int]:
    """Return ``(covered_question_instances, target_instances)`` for rows listed in the aggregate.

    Uses the same rule as merge dedupe: latest ``date_utc`` per ``(seed, sample_size)``.
    That JSON must contain a ``heatmap`` block to count toward *covered*.
    """
    target = sum(int(r["sample_size"]) for r in target_rows)
    groups: dict[tuple[int, int], list[tuple[str, dict[str, Any]]]] = defaultdict(
        list
    )
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


def _best_grid_cell(
    grid: np.ndarray,
    n_hops: list[int],
    tau: list[float],
) -> dict[str, Any]:
    g = np.asarray(grid, dtype=np.float64)
    if g.size == 0 or not np.isfinite(g).any():
        return {"n_hops": None, "tau": None, "value": float("nan")}
    g2 = np.where(np.isfinite(g), g, -np.inf)
    idx = int(np.argmax(g2))
    ii, jj = np.unravel_index(idx, g2.shape)
    return {
        "n_hops": int(n_hops[ii]),
        "tau": float(tau[jj]),
        "value": float(g[ii, jj]),
    }


def _merge_weighted(
    runs: list[tuple[Path, dict[str, Any]]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    list[float],
    list[dict[str, Any]],
    int,
]:
    if not runs:
        raise SystemExit("No daily_runs/*.json files contain a 'heatmap' key.")

    n_hops = None
    tau = None
    sum_f1: np.ndarray | None = None
    sum_pk: np.ndarray | None = None
    total_w = 0
    meta: list[dict[str, Any]] = []

    for path, data in runs:
        hm = data["heatmap"]
        nh = list(hm["n_hops"])
        tv = [float(x) for x in hm["tau"]]
        if n_hops is None:
            n_hops = nh
            tau = tv
            g1 = np.asarray(hm["mean_f1_grid"], dtype=np.float64)
            g2 = np.asarray(hm["mean_retrieval_p_at_k_grid"], dtype=np.float64)
            if g1.shape != (len(n_hops), len(tau)) or g2.shape != g1.shape:
                raise SystemExit(f"Bad grid shape in {path}")
            sum_f1 = np.zeros_like(g1)
            sum_pk = np.zeros_like(g2)
        elif nh != n_hops or tv != tau:
            raise SystemExit(
                f"n_hops/tau mismatch: {path.name} vs first file "
                f"({nh}/{tv} vs {n_hops}/{tau})"
            )

        w = int(data.get("sample_size", 0))
        if w <= 0:
            raise SystemExit(f"Invalid sample_size in {path}")

        g1 = np.asarray(hm["mean_f1_grid"], dtype=np.float64)
        g2 = np.asarray(hm["mean_retrieval_p_at_k_grid"], dtype=np.float64)
        assert sum_f1 is not None and sum_pk is not None
        sum_f1 += g1 * w
        sum_pk += g2 * w
        total_w += w
        meta.append(
            {
                "path": str(path.relative_to(REPO)).replace("\\", "/"),
                "seed": data.get("seed"),
                "sample_size": w,
                "date_utc": data.get("date_utc"),
            }
        )

    assert sum_f1 is not None and sum_pk is not None and n_hops is not None and tau is not None
    if total_w <= 0:
        raise SystemExit("total weight 0")
    mean_f1 = sum_f1 / total_w
    mean_pk = sum_pk / total_w
    return mean_f1, mean_pk, n_hops, tau, meta, total_w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--daily-dir",
        type=Path,
        default=REPO / "results" / "daily_runs",
        help="Directory with per-seed JSON (top-level *.json only)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results" / "merged_heatmap",
    )
    ap.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Include every heatmap JSON (may double-count same seed if re-run)",
    )
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    runs = _load_runs(args.daily_dir)
    if not args.no_dedupe:
        runs = _dedupe_latest_date(runs)
    mean_f1, mean_pk, n_hops, tau, meta, total_w = _merge_weighted(runs)

    best_f1 = _best_grid_cell(mean_f1, n_hops, tau)
    best_pk = _best_grid_cell(mean_pk, n_hops, tau)
    payload = {
        "n_source_files": len(runs),
        "total_question_instances_weighted": total_w,
        "n_hops": n_hops,
        "tau": tau,
        "mean_f1_grid": mean_f1.tolist(),
        "mean_retrieval_p_at_k_grid": mean_pk.tolist(),
        "best_mean_answer_f1": best_f1,
        "best_mean_retrieval_p_at_k": best_pk,
        "sources": meta,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "merged_heatmap.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json} ({len(runs)} files, total weight={total_w})")

    if not args.no_plots:
        fungi_heatmap(
            mean_f1,
            n_hops,
            tau,
            args.out_dir / "heatmap_merged_mean_f1.png",
            title="SP-GQE: merged mean answer F1 (n × τ), weighted across seeds",
            cbar_label="Mean answer F1",
        )
        fungi_heatmap(
            mean_pk,
            n_hops,
            tau,
            args.out_dir / "heatmap_merged_mean_p_at_k.png",
            title="SP-GQE: merged mean retrieval P@k (n × τ), weighted across seeds",
            cbar_label="Mean P@k",
        )
        print(f"Wrote plots under {args.out_dir}")

    # Compact table for stdout
    print("\nMerged mean F1 grid (rows=n_hops, cols=tau):")
    print(np.round(mean_f1, 4))
    print("\nMerged mean P@k grid:")
    print(np.round(mean_pk, 4))

    # Avoid Greek tau in stdout: Windows cp1252 consoles raise UnicodeEncodeError.
    print("\n## Best (n, tau) on merged grids (weighted by sample_size)\n")
    print("| Metric | n_hops | tau | Value |")
    print("|--------|--------|---|-------|")
    bf, bpk = best_f1, best_pk
    print(
        f"| Mean answer F1 | {bf['n_hops']} | {bf['tau']} | "
        f"{bf['value']:.4f} |"
    )
    print(
        f"| Mean retrieval P@k | {bpk['n_hops']} | {bpk['tau']} | "
        f"{bpk['value']:.4f} |"
    )


if __name__ == "__main__":
    main()
