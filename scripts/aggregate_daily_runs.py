#!/usr/bin/env python3
"""Aggregate per-seed runs under results/daily_runs/ into a final summary.

Each file is expected to be produced by scripts/run_experiment.py and contain
the `pipelines`, `paired_deltas`, `graph_query_log`, and `graph_query_validity`
keys. Re-running the aggregator after adding more daily seeds is idempotent —
it reads whatever is currently in results/daily_runs/.

By default, files are **deduped** to one row per ``(seed, sample_size)``, keeping
the latest ``date_utc``, so re-runs replace older JSON for the same seed.

Outputs:
  results/AGGREGATED_SUMMARY.json   machine-readable
  results/AGGREGATED_REPORT.md      human-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]

PIPELINE_KEYS = [
    "V-RAG",
    "GQE-RAG(n=2)",
    "SP-GQE(n=2,τ=0.5)",
    "SP-GQE-i(n=3,τ=0.5)",
    "GR-RAG",
    "GF-RAG",
]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _mean_ci_t(values: list[float]) -> tuple[float, tuple[float, float]]:
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    m = float(np.mean(a))
    if n == 1:
        return m, (m, m)
    sem = stats.sem(a, ddof=1)
    h = sem * stats.t.ppf(0.975, df=n - 1)
    return m, (m - h, m + h)


def _dedupe_latest_run_dict(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per ``(seed, sample_size)``: keep the latest ``date_utc`` (re-runs)."""
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        sk = int(run.get("seed", -1))
        n = int(run.get("sample_size", 0))
        groups[(sk, n)].append(run)
    picked: list[dict[str, Any]] = []
    for key in sorted(groups):
        items = groups[key]
        items.sort(key=lambda r: str(r.get("date_utc", "")))
        picked.append(items[-1])
    return picked


def _bootstrap_ci_mean_diff(
    deltas: list[float], seed: int = 42, n_boot: int = 8000
) -> tuple[float, tuple[float, float]]:
    a = np.asarray(deltas, dtype=np.float64)
    if len(a) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(a)
    for i in range(n_boot):
        means[i] = rng.choice(a, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(a)), (float(lo), float(hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--daily-dir",
        type=Path,
        default=REPO / "results" / "daily_runs",
        help="Directory with per-seed run JSON files",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Use every JSON file (may double-count the same seed if re-run on different dates)",
    )
    args = ap.parse_args()

    files = sorted(args.daily_dir.glob("*.json"))
    if not files:
        print(f"No per-seed JSON files in {args.daily_dir}", file=sys.stderr)
        sys.exit(1)

    runs_raw: list[dict[str, Any]] = []
    for p in files:
        try:
            runs_raw.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[skip] {p.name}: {e}", file=sys.stderr)

    runs = runs_raw if args.no_dedupe else _dedupe_latest_run_dict(runs_raw)
    if len(runs) < len(runs_raw):
        print(
            f"Deduped {len(runs_raw)} files → {len(runs)} unique (seed, sample_size) runs "
            "(latest date_utc kept).",
            file=sys.stderr,
        )

    seed_means_f1: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    seed_means_em: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    seed_means_sup: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    seed_means_pk: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    bridge_deltas: list[float] = []
    comp_deltas: list[float] = []
    gv_b1_p: list[float] = []
    gv_b1_r: list[float] = []
    gv_b2_p: list[float] = []
    gv_b2_r: list[float] = []
    gv_union_p: list[float] = []
    gv_union_r: list[float] = []
    gv_kept_p: list[float] = []
    gv_kept_r: list[float] = []
    total_instances = 0
    seeds_used: list[dict[str, Any]] = []

    for run in runs:
        seeds_used.append(
            {
                "date_utc": run.get("date_utc"),
                "seed": run.get("seed"),
                "sample_size": run.get("sample_size"),
                "llm_backend": run.get("llm_backend"),
            }
        )
        total_instances += int(run.get("sample_size", 0) or 0)
        pipelines = run.get("pipelines", {}) or {}
        for k in PIPELINE_KEYS:
            if k in pipelines:
                seed_means_f1[k].append(float(pipelines[k].get("mean_f1", 0.0)))
                seed_means_em[k].append(float(pipelines[k].get("mean_em", 0.0)))
                seed_means_sup[k].append(
                    float(pipelines[k].get("mean_supporting_title_recall_at_k", 0.0))
                )
                seed_means_pk[k].append(
                    float(pipelines[k].get("mean_retrieval_p_at_k", 0.0))
                )
        pd = run.get("paired_deltas", {}) or {}
        for row in pd.get("bridge", []) or []:
            bridge_deltas.append(float(row.get("delta_f1", 0.0)))
        for row in pd.get("comparison", []) or []:
            comp_deltas.append(float(row.get("delta_f1", 0.0)))
        for entry in run.get("graph_query_log", []) or []:
            v = entry.get("validity", {}) or {}
            gv_b1_p.append(float(v.get("graph_precision_branch1", 0.0)))
            gv_b1_r.append(float(v.get("graph_recall_branch1", 0.0)))
            gv_b2_p.append(float(v.get("graph_precision_branch2", 0.0)))
            gv_b2_r.append(float(v.get("graph_recall_branch2", 0.0)))
            gv_union_p.append(float(v.get("graph_precision_union", 0.0)))
            gv_union_r.append(float(v.get("graph_recall_union", 0.0)))
            gv_kept_p.append(float(v.get("graph_precision_kept", 0.0)))
            gv_kept_r.append(float(v.get("graph_recall_kept", 0.0)))

    aggregated: dict[str, Any] = {}
    for k in PIPELINE_KEYS:
        m_f1, ci_f1 = _mean_ci_t(seed_means_f1[k])
        m_em, ci_em = _mean_ci_t(seed_means_em[k])
        m_sr, ci_sr = _mean_ci_t(seed_means_sup[k])
        m_pk, ci_pk = _mean_ci_t(seed_means_pk[k])
        aggregated[k] = {
            "mean_f1": m_f1,
            "ci95_f1": list(ci_f1),
            "mean_em": _clip01(m_em),
            "ci95_em": [_clip01(ci_em[0]), _clip01(ci_em[1])],
            "mean_supporting_title_recall_at_k": _clip01(m_sr),
            "ci95_supporting_title_recall": [_clip01(ci_sr[0]), _clip01(ci_sr[1])],
            "mean_retrieval_p_at_k": _clip01(m_pk),
            "ci95_retrieval_p_at_k": [_clip01(ci_pk[0]), _clip01(ci_pk[1])],
            "per_seed_mean_f1": seed_means_f1[k],
            "n_seeds": len(seed_means_f1[k]),
        }

    bd_m, bd_ci = _bootstrap_ci_mean_diff(bridge_deltas)
    cd_m, cd_ci = _bootstrap_ci_mean_diff(comp_deltas)

    def _mean_or_nan(x: list[float]) -> float:
        return float(np.mean(x)) if x else float("nan")

    summary: dict[str, Any] = {
        "n_files": len(runs),
        "total_question_instances": total_instances,
        "seeds_used": seeds_used,
        "aggregated_across_seeds": aggregated,
        "protocol_default_n2_tau05": {
            "paired_delta_f1_SP_GQE_minus_V_RAG": {
                "bridge": {
                    "mean": bd_m,
                    "bootstrap_ci95": list(bd_ci),
                    "n_pairs": len(bridge_deltas),
                },
                "comparison": {
                    "mean": cd_m,
                    "bootstrap_ci95": list(cd_ci),
                    "n_pairs": len(comp_deltas),
                },
            },
            "graph_query_validity_pooled": {
                "branch1_nhop": {
                    "mean_precision": _mean_or_nan(gv_b1_p),
                    "mean_recall": _mean_or_nan(gv_b1_r),
                },
                "branch2_keyword": {
                    "mean_precision": _mean_or_nan(gv_b2_p),
                    "mean_recall": _mean_or_nan(gv_b2_r),
                },
                "union": {
                    "mean_precision": _mean_or_nan(gv_union_p),
                    "mean_recall": _mean_or_nan(gv_union_r),
                },
                "kept_after_tau": {
                    "mean_precision": _mean_or_nan(gv_kept_p),
                    "mean_recall": _mean_or_nan(gv_kept_r),
                },
                "n_questions": len(gv_b1_p),
            },
        },
    }

    merged_path = args.out_dir / "merged_heatmap" / "merged_heatmap.json"
    merged_block: dict[str, Any] | None = None
    if merged_path.is_file():
        try:
            from heatmap_merged_config import compute_merged_config_aggregates

            merged_block = compute_merged_config_aggregates(
                runs,
                merged_path,
                mean_ci_t=_mean_ci_t,
                bootstrap_ci_mean_diff=_bootstrap_ci_mean_diff,
                clip01=_clip01,
            )
            summary["merged_config_aggregates"] = merged_block
        except Exception as e:
            summary["merged_config_aggregates_error"] = str(e)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "AGGREGATED_SUMMARY.json"
    out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines: list[str] = [
        f"# Aggregated results — {len(runs)} per-seed files",
        "",
        f"- **Total question instances:** {total_instances}",
        f"- **Seeds aggregated:** "
        + ", ".join(
            f"{s['date_utc']}:seed{s['seed']}(n={s['sample_size']})" for s in seeds_used
        ),
        "",
        "## Per-pipeline (mean ± 95% CI across seed-level means)",
        "",
        "| Pipeline | Mean F1 | 95% CI F1 | Mean EM | Mean Sup-Title Recall@k | Mean P@k | n seeds |",
        "|----------|---------|-----------|---------|-------------------------|----------|---------|",
    ]
    for k in PIPELINE_KEYS:
        a = aggregated[k]
        lines.append(
            f"| {k} | {a['mean_f1']:.4f} | "
            f"[{a['ci95_f1'][0]:.4f}, {a['ci95_f1'][1]:.4f}] | "
            f"{a['mean_em']:.4f} | "
            f"{a['mean_supporting_title_recall_at_k']:.4f} | "
            f"{a['mean_retrieval_p_at_k']:.4f} | "
            f"{a['n_seeds']} |"
        )

    if merged_block:
        from heatmap_merged_config import (
            format_per_pipeline_row,
            markdown_sections,
            per_pipeline_rows_at_merged_config,
        )

        sel = merged_block["selection"]
        sp_n = merged_block["sp_gqe_at_selected_config"]["n_seeds"]
        lines += [
            "",
            f"*{sel['n_hops']}, τ={sel['tau']} below: **post-hoc argmax** on pooled "
            f"heatmap mean F1 (not a pre-registered protocol arm). EM / sup-title recall "
            f"not swept in heatmap runs (—). n={sp_n} heatmap seeds. SP-GQE-i row needs "
            f"``heatmap_sp_gqe_i`` in daily JSON (added in newer ``run_experiment.py`` "
            f"runs).*",
            "",
        ]
        for label, row in per_pipeline_rows_at_merged_config(merged_block):
            lines.append(format_per_pipeline_row(label, row))

        lines.extend(["", *markdown_sections(merged_block)])
    else:
        lines += [
            "",
            "## SP-GQE at merged-grid selected (n, τ)",
            "",
            "*(No `merged_heatmap/merged_heatmap.json` — run `merge_heatmaps.py` first.)*",
            "",
            "## Paired ΔF1 / graph-query validity",
            "",
            "*(Merged-config sections require merged heatmap.)*",
            "",
        ]

    lines += [
        "",
        "## Appendix: protocol default (n=2, τ=0.5)",
        "",
        "Logged during experiments before merged-grid selection was applied to "
        "reporting. Use merged-config sections above as primary when available.",
        "",
        "### Paired ΔF1 (SP-GQE(n=2, τ=0.5) − V-RAG), question-level",
        "",
        "| Subset | Mean Δ | Bootstrap 95% CI | n pairs |",
        "|--------|--------|------------------|---------|",
        f"| bridge | {bd_m:.4f} | [{bd_ci[0]:.4f}, {bd_ci[1]:.4f}] | {len(bridge_deltas)} |",
        f"| comparison | {cd_m:.4f} | [{cd_ci[0]:.4f}, {cd_ci[1]:.4f}] | {len(comp_deltas)} |",
        "",
        "### Graph-query validity (n=2, τ=0.5 log)",
        "",
        "| Stage | Mean precision | Mean recall | n questions |",
        "|-------|----------------|-------------|-------------|",
        f"| Branch 1 (n-hop) | {_mean_or_nan(gv_b1_p):.4f} | {_mean_or_nan(gv_b1_r):.4f} | {len(gv_b1_p)} |",
        f"| Branch 2 (keyword) | {_mean_or_nan(gv_b2_p):.4f} | {_mean_or_nan(gv_b2_r):.4f} | {len(gv_b2_p)} |",
        f"| Union | {_mean_or_nan(gv_union_p):.4f} | {_mean_or_nan(gv_union_r):.4f} | {len(gv_union_p)} |",
        f"| Kept after τ=0.5 | {_mean_or_nan(gv_kept_p):.4f} | {_mean_or_nan(gv_kept_r):.4f} | {len(gv_kept_p)} |",
        "",
    ]

    try:
        from build_heatmap_tuned_comparison import build_comparison

        tuned_payload, tuned_md = build_comparison(
            runs,
            merged_heatmap_path=merged_path if merged_path.is_file() else None,
        )
        (args.out_dir / "heatmap_tuned_comparison.json").write_text(
            json.dumps(tuned_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (args.out_dir / "EXPERIMENT_REPORT_PIPELINE_COMPARISON.md").write_text(
            "\n".join(tuned_md), encoding="utf-8"
        )
        print(
            f"Wrote {args.out_dir / 'heatmap_tuned_comparison.json'} and "
            f"{args.out_dir / 'EXPERIMENT_REPORT_PIPELINE_COMPARISON.md'}"
        )
    except Exception as e:
        print(f"[heatmap comparison] {e}", file=sys.stderr)

    out_md = args.out_dir / "AGGREGATED_REPORT.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
