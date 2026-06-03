#!/usr/bin/env python3
"""Pipeline comparison table using merged-grid-selected (n, tau) for SP-GQE.

Uses one global (n*, tau*) from ``merged_heatmap.json`` (argmax pooled F1 grid).
Per-seed SP-GQE metrics are read at that cell — not per-seed argmax.

See ``heatmap_merged_config.py`` for paired ΔF1 and graph-validity rules.

Outputs:
  results/heatmap_tuned_comparison.json
  results/EXPERIMENT_REPORT_PIPELINE_COMPARISON.md
  results/pipelines_bar_f1_heatmap_tuned.png  (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import aggregate_daily_runs as agg  # noqa: E402
from heatmap_merged_config import (  # noqa: E402
    compute_merged_config_aggregates,
    markdown_sections,
)

PIPELINE_KEYS = agg.PIPELINE_KEYS
_clip01 = agg._clip01
_dedupe_latest_run_dict = agg._dedupe_latest_run_dict
_mean_ci_t = agg._mean_ci_t
_bootstrap_ci_mean_diff = agg._bootstrap_ci_mean_diff

try:
    from sp_gqe.experiment.plots import bar_comparison  # noqa: E402
except ImportError:
    bar_comparison = None  # type: ignore[misc, assignment]


def build_comparison(
    runs: list[dict[str, Any]],
    *,
    merged_heatmap_path: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    seed_means: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    seed_means_pk: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    for run in runs:
        pl = run.get("pipelines", {}) or {}
        for k in PIPELINE_KEYS:
            if k in pl:
                seed_means[k].append(float(pl[k].get("mean_f1", 0.0)))
                seed_means_pk[k].append(float(pl[k].get("mean_retrieval_p_at_k", 0.0)))

    rows_out: list[dict[str, Any]] = []
    for label, key in [
        ("V-RAG", "V-RAG"),
        ("GQE-RAG(n=2)", "GQE-RAG(n=2)"),
        ("GR-RAG", "GR-RAG"),
        ("GF-RAG", "GF-RAG"),
        ("SP-GQE (fixed n=2, τ=0.5)", "SP-GQE(n=2,τ=0.5)"),
        ("SP-GQE-i (fixed n=3, τ=0.5)", "SP-GQE-i(n=3,τ=0.5)"),
    ]:
        if seed_means[key]:
            m, ci = _mean_ci_t(seed_means[key])
            mp, cip = _mean_ci_t(seed_means_pk[key])
            rows_out.append(
                {
                    "pipeline": label,
                    "mean_f1": m,
                    "ci95_f1": list(ci),
                    "mean_p_at_k": _clip01(mp),
                    "ci95_p_at_k": [_clip01(cip[0]), _clip01(cip[1])],
                    "n_seeds": len(seed_means[key]),
                }
            )

    merged_block: dict[str, Any] | None = None
    if merged_heatmap_path and merged_heatmap_path.is_file():
        merged_block = compute_merged_config_aggregates(
            runs,
            merged_heatmap_path,
            mean_ci_t=_mean_ci_t,
            bootstrap_ci_mean_diff=_bootstrap_ci_mean_diff,
            clip01=_clip01,
        )
        from heatmap_merged_config import per_pipeline_rows_at_merged_config

        for label, row in per_pipeline_rows_at_merged_config(merged_block):
            if row.get("pending_heatmap_sp_gqe_i"):
                continue
            ci_pk = row.get("ci95_retrieval_p_at_k")
            if ci_pk is None:
                mp = row["mean_retrieval_p_at_k"]
                ci_pk = [mp, mp]
            rows_out.append(
                {
                    "pipeline": label,
                    "mean_f1": row["mean_f1"],
                    "ci95_f1": row["ci95_f1"],
                    "mean_p_at_k": row["mean_retrieval_p_at_k"],
                    "ci95_p_at_k": ci_pk,
                    "n_seeds": row["n_seeds"],
                }
            )

    payload: dict[str, Any] = {
        "comparison_rows": rows_out,
        "merged_config_aggregates": merged_block,
    }

    md: list[str] = [
        "# Pipeline comparison (merged-grid SP-GQE config)",
        "",
        "SP-GQE / SP-GQE-i at **one** (n, τ) from the pooled heatmap F1 argmax. "
        "Per-seed values use that same cell. SP-GQE-i requires ``heatmap_sp_gqe_i`` "
        "in daily JSON (newer runs).",
        "",
        "## Comparison table (mean token F1 and retrieval P@k)",
        "",
        "| Pipeline | Mean F1 | 95% CI F1 | Mean P@k | 95% CI P@k | n seeds |",
        "|----------|---------|-----------|----------|------------|---------|",
    ]
    for r in rows_out:
        md.append(
            f"| {r['pipeline']} | {r['mean_f1']:.4f} | "
            f"[{r['ci95_f1'][0]:.4f}, {r['ci95_f1'][1]:.4f}] | "
            f"{r['mean_p_at_k']:.4f} | "
            f"[{r['ci95_p_at_k'][0]:.4f}, {r['ci95_p_at_k'][1]:.4f}] | "
            f"{r['n_seeds']} |"
        )
    if merged_block:
        md.extend(["", *markdown_sections(merged_block)])
    md.append("")
    return payload, md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--daily-dir", type=Path, default=REPO / "results" / "daily_runs")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument(
        "--merged-heatmap",
        type=Path,
        default=REPO / "results" / "merged_heatmap" / "merged_heatmap.json",
    )
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    runs_raw: list[dict[str, Any]] = []
    for p in sorted(args.daily_dir.glob("*.json")):
        try:
            runs_raw.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip] {p.name}: {e}", file=sys.stderr)

    runs = runs_raw if args.no_dedupe else _dedupe_latest_run_dict(runs_raw)
    merged_path = args.merged_heatmap if args.merged_heatmap.is_file() else None
    payload, md_lines = build_comparison(runs, merged_heatmap_path=merged_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "heatmap_tuned_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.out_dir / "EXPERIMENT_REPORT_PIPELINE_COMPARISON.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    print(f"Wrote {args.out_dir / 'heatmap_tuned_comparison.json'}")
    print(f"Wrote {args.out_dir / 'EXPERIMENT_REPORT_PIPELINE_COMPARISON.md'}")

    if not args.no_plot and bar_comparison is not None:
        plot_rows = [r for r in payload["comparison_rows"] if r.get("ci95_f1")]
        bar_comparison(
            [r["pipeline"] for r in plot_rows],
            [r["mean_f1"] for r in plot_rows],
            args.out_dir / "pipelines_bar_f1_heatmap_tuned.png",
        )
        print(f"Wrote {args.out_dir / 'pipelines_bar_f1_heatmap_tuned.png'}")


if __name__ == "__main__":
    main()
