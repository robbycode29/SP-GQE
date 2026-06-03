"""Stats at the merged-heatmap-selected (n, tau) — one global config for all seeds.

Selection: argmax on the **sample-size-weighted** ``mean_f1_grid`` in
``merged_heatmap.json`` (same rule as ``merge_heatmaps._best_grid_cell``).

Per-seed metrics: read that seed's ``heatmap`` cell at ``(n*, tau*)`` (not per-seed
argmax). Aggregate mean F1 / P@k with t-based 95% CI across seeds.

Paired ΔF1: question-level bridge / comparison (bootstrap over pairs) from
``paired_deltas`` on heatmap seeds; seed-level overall row uses heatmap cell F1
at ``(n*, tau*)`` minus V-RAG. When ``(n*, tau*) != (2, 0.5)``, pair-level
``sp_gqe_f1`` in logs is still at protocol (n=2, τ=0.5) — noted in the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from merge_heatmaps import _best_grid_cell  # noqa: E402


def _heatmap_cell(
    run: dict[str, Any],
    n_hops: int,
    tau: float,
    *,
    heatmap_key: str = "heatmap",
) -> tuple[float, float] | None:
    hm = run.get(heatmap_key)
    if not isinstance(hm, dict):
        return None
    nh = list(hm["n_hops"])
    tv = [float(x) for x in hm["tau"]]
    try:
        ii = nh.index(int(n_hops))
        jj = tv.index(float(tau))
    except ValueError:
        return None
    f1g = np.asarray(hm["mean_f1_grid"], dtype=np.float64)
    pkg = np.asarray(hm["mean_retrieval_p_at_k_grid"], dtype=np.float64)
    if f1g.shape != (len(nh), len(tv)):
        return None
    return float(f1g[ii, jj]), float(pkg[ii, jj])


def load_merged_f1_best_config(merged_path: Path) -> tuple[int, float, dict[str, Any]]:
    mh = json.loads(merged_path.read_text(encoding="utf-8"))
    nh = list(mh["n_hops"])
    tv = [float(x) for x in mh["tau"]]
    f1g = np.asarray(mh["mean_f1_grid"], dtype=np.float64)
    bf = _best_grid_cell(f1g, nh, tv)
    if bf["n_hops"] is None or bf["tau"] is None:
        raise ValueError("merged heatmap has no finite F1 best cell")
    return int(bf["n_hops"]), float(bf["tau"]), mh


def compute_merged_config_aggregates(
    runs: list[dict[str, Any]],
    merged_path: Path,
    *,
    mean_ci_t: Any,
    bootstrap_ci_mean_diff: Any,
    clip01: Any,
) -> dict[str, Any]:
    """Return JSON-serialisable block for AGGREGATED_SUMMARY / reports."""
    n_star, tau_star, mh = load_merged_f1_best_config(merged_path)
    nh = list(mh["n_hops"])
    tv = [float(x) for x in mh["tau"]]
    f1g = np.asarray(mh["mean_f1_grid"], dtype=np.float64)
    pkg = np.asarray(mh["mean_retrieval_p_at_k_grid"], dtype=np.float64)
    ii = nh.index(n_star)
    jj = tv.index(tau_star)
    pooled_f1 = float(f1g[ii, jj])
    pooled_pk = float(pkg[ii, jj])

    seed_f1: list[float] = []
    seed_pk: list[float] = []
    seed_f1_i: list[float] = []
    seed_pk_i: list[float] = []
    seed_vr: list[float] = []
    seed_delta: list[float] = []
    per_seed_rows: list[dict[str, Any]] = []

    bridge_deltas_q: list[float] = []
    comp_deltas_q: list[float] = []
    gv_b1_p: list[float] = []
    gv_b1_r: list[float] = []
    gv_b2_p: list[float] = []
    gv_b2_r: list[float] = []
    gv_union_p: list[float] = []
    gv_union_r: list[float] = []
    gv_kept_p: list[float] = []
    gv_kept_r: list[float] = []

    protocol_default = n_star == 2 and abs(tau_star - 0.5) < 1e-9

    for run in runs:
        cell = _heatmap_cell(run, n_star, tau_star, heatmap_key="heatmap")
        if cell is None:
            continue
        f1_sp, pk_sp = cell
        cell_i = _heatmap_cell(run, n_star, tau_star, heatmap_key="heatmap_sp_gqe_i")
        if cell_i is not None:
            f1_i, pk_i = cell_i
            seed_f1_i.append(f1_i)
            seed_pk_i.append(pk_i)
        pl = run.get("pipelines", {}) or {}
        f1_vr = float(pl.get("V-RAG", {}).get("mean_f1", 0.0))
        seed_f1.append(f1_sp)
        seed_pk.append(pk_sp)
        seed_vr.append(f1_vr)
        seed_delta.append(f1_sp - f1_vr)
        per_seed_rows.append(
            {
                "seed": run.get("seed"),
                "sample_size": run.get("sample_size"),
                "date_utc": run.get("date_utc"),
                "mean_f1_sp_gqe": f1_sp,
                "mean_f1_v_rag": f1_vr,
                "delta_f1": f1_sp - f1_vr,
                "mean_p_at_k": pk_sp,
            }
        )

        pd = run.get("paired_deltas", {}) or {}
        for row in pd.get("bridge", []) or []:
            bridge_deltas_q.append(float(row.get("delta_f1", 0.0)))
        for row in pd.get("comparison", []) or []:
            comp_deltas_q.append(float(row.get("delta_f1", 0.0)))

        if protocol_default:
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

    def _mean_or_nan(x: list[float]) -> float:
        return float(np.mean(x)) if x else float("nan")

    m_f1, ci_f1 = mean_ci_t(seed_f1)
    m_pk, ci_pk = mean_ci_t(seed_pk)
    m_d, ci_d = mean_ci_t(seed_delta)
    sp_i_block: dict[str, Any] | None = None
    if seed_f1_i:
        m_f1_i, ci_f1_i = mean_ci_t(seed_f1_i)
        m_pk_i, ci_pk_i = mean_ci_t(seed_pk_i)
        sp_i_block = {
            "mean_f1": m_f1_i,
            "ci95_f1": list(ci_f1_i),
            "mean_retrieval_p_at_k": clip01(m_pk_i),
            "ci95_retrieval_p_at_k": [clip01(ci_pk_i[0]), clip01(ci_pk_i[1])],
            "n_seeds": len(seed_f1_i),
        }

    out: dict[str, Any] = {
        "selection": {
            "criterion": "argmax on merged sample-size-weighted mean_f1_grid",
            "n_hops": n_star,
            "tau": tau_star,
            "pooled_mean_f1_on_merged_grid": pooled_f1,
            "pooled_mean_p_at_k_on_merged_grid": pooled_pk,
            "merged_total_weight": mh.get("total_question_instances_weighted"),
            "merged_n_source_files": mh.get("n_source_files"),
        },
        "sp_gqe_at_selected_config": {
            "mean_f1": m_f1,
            "ci95_f1": list(ci_f1),
            "mean_retrieval_p_at_k": clip01(m_pk),
            "ci95_retrieval_p_at_k": [clip01(ci_pk[0]), clip01(ci_pk[1])],
            "n_seeds": len(seed_f1),
            "per_seed": per_seed_rows,
        },
        "sp_gqe_i_at_selected_config": sp_i_block,
        "paired_delta_f1_seed_level_SP_GQE_minus_V_RAG": {
            "mean_delta_f1": m_d,
            "ci95_delta_f1": list(ci_d),
            "n_seeds": len(seed_delta),
            "note": (
                "Per-seed delta = heatmap cell F1 at (n*,tau*) minus V-RAG pipeline "
                "mean F1 on the same questions. Not question-level bootstrap."
            ),
        },
        "paired_delta_f1_question_level": None,
        "graph_query_validity_pooled": None,
        "logging_matches_selected_config": protocol_default,
    }

    if bridge_deltas_q or comp_deltas_q:
        bd_m, bd_ci = bootstrap_ci_mean_diff(bridge_deltas_q)
        cd_m, cd_ci = bootstrap_ci_mean_diff(comp_deltas_q)
        out["paired_delta_f1_question_level"] = {
            "sp_gqe_arm_in_paired_deltas": "n=2, tau=0.5 (protocol default in daily_runs)",
            "matches_merged_selection": protocol_default,
            "heatmap_seed_questions_only": True,
            "bridge": {
                "mean_delta_f1": bd_m,
                "bootstrap_ci95": list(bd_ci),
                "n_pairs": len(bridge_deltas_q),
            },
            "comparison": {
                "mean_delta_f1": cd_m,
                "bootstrap_ci95": list(cd_ci),
                "n_pairs": len(comp_deltas_q),
            },
        }
    if protocol_default:
        out["graph_query_validity_pooled"] = {
            "config": "n=2, tau=0.5 (matches experiment log)",
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
        }
    if not protocol_default:
        out["paired_delta_f1_note"] = (
            f"Merged-grid selected (n={n_star}, tau={tau_star}). Bridge/comparison "
            "rows pool question-level ΔF1 from heatmap seeds where "
            "SP-GQE in each pair is still SP-GQE(n=2, τ=0.5) from the experiment log; "
            "only the seed-level row uses SP-GQE at (n*, τ*) from the heatmap cell."
        )
        out["graph_query_validity_note"] = (
            f"Graph-query validity below is recomputed from ``graph_query_log`` at "
            f"τ={tau_star} (cosine prune on stored similarities). Branch 1 n-hop "
            f"structure in the log was captured at n=2 during the run; merged "
            f"selection n={n_star} may differ — see validity section."
        )

    return out


def markdown_sections(block: dict[str, Any]) -> list[str]:
    """Markdown lines for AGGREGATED_REPORT (no leading H1)."""
    sel = block["selection"]
    sp = block["sp_gqe_at_selected_config"]
    pd_seed = block["paired_delta_f1_seed_level_SP_GQE_minus_V_RAG"]
    n_star, tau_star = sel["n_hops"], sel["tau"]

    lines = [
        "## SP-GQE at merged-grid selected (n, τ)",
        "",
        f"**Selection:** {sel['criterion']}. "
        f"Global best on the pooled heatmap: **n={n_star}, τ={tau_star}** "
        f"(pooled mean F1 = {sel['pooled_mean_f1_on_merged_grid']:.4f}, "
        f"P@k = {sel['pooled_mean_p_at_k_on_merged_grid']:.4f}; "
        f"weight = {sel.get('merged_total_weight')}, "
        f"{sel.get('merged_n_source_files')} heatmap files).",
        "",
        "Per-seed metrics use **that same (n, τ)** cell from each seed's stored "
        "heatmap (not per-seed argmax).",
        "",
        "| Metric | Mean | 95% CI | n seeds |",
        "|--------|------|--------|---------|",
        f"| Mean answer F1 | {sp['mean_f1']:.4f} | "
        f"[{sp['ci95_f1'][0]:.4f}, {sp['ci95_f1'][1]:.4f}] | {sp['n_seeds']} |",
        f"| Mean retrieval P@k | {sp['mean_retrieval_p_at_k']:.4f} | "
        f"[{sp['ci95_retrieval_p_at_k'][0]:.4f}, {sp['ci95_retrieval_p_at_k'][1]:.4f}] | "
        f"{sp['n_seeds']} |",
        "",
        "## Paired ΔF1 (SP-GQE − V-RAG)",
        "",
        f"Heatmap-selected config: **n={n_star}, τ={tau_star}**. "
        "Question-level rows pool pairs from **heatmap seeds only**.",
        "",
        "| Subset | Mean Δ | 95% CI | n |",
        "|--------|--------|--------|---|",
    ]

    pq = block.get("paired_delta_f1_question_level")
    if pq:
        lines += [
            f"| bridge | {pq['bridge']['mean_delta_f1']:.4f} | "
            f"bootstrap [{pq['bridge']['bootstrap_ci95'][0]:.4f}, "
            f"{pq['bridge']['bootstrap_ci95'][1]:.4f}] | {pq['bridge']['n_pairs']} pairs |",
            f"| comparison | {pq['comparison']['mean_delta_f1']:.4f} | "
            f"bootstrap [{pq['comparison']['bootstrap_ci95'][0]:.4f}, "
            f"{pq['comparison']['bootstrap_ci95'][1]:.4f}] | {pq['comparison']['n_pairs']} pairs |",
        ]
    else:
        lines += [
            "| bridge | — | — | 0 pairs |",
            "| comparison | — | — | 0 pairs |",
        ]

    lines += [
        f"| all (seed-level; SP-GQE n={n_star}, τ={tau_star} vs V-RAG) | "
        f"{pd_seed['mean_delta_f1']:.4f} | "
        f"t [{pd_seed['ci95_delta_f1'][0]:.4f}, {pd_seed['ci95_delta_f1'][1]:.4f}] | "
        f"{pd_seed['n_seeds']} seeds |",
        "",
    ]

    if pq and not pq.get("matches_merged_selection"):
        lines += [
            f"*Bridge/comparison pairs use SP-GQE from the experiment log "
            f"({pq.get('sp_gqe_arm_in_paired_deltas')}), not SP-GQE(n={n_star}, τ={tau_star}), "
            "because per-question F1 at the heatmap-selected cell was not stored. "
            "The **all (seed-level)** row uses the heatmap cell at (n*, τ*).*",
            "",
        ]
    elif pq and pq.get("matches_merged_selection"):
        lines += [
            "*Bridge/comparison and seed-level rows all use SP-GQE(n=2, τ=0.5), "
            "matching the merged-grid selection.*",
            "",
        ]

    gv = block.get("graph_query_validity_pooled")
    if gv:
        lines += [
            "## Graph-query validity (pooled per question, selected config)",
            "",
            f"Logged at **{gv['config']}** (matches merged-grid selection).",
            "",
            "| Stage | Mean precision | Mean recall | n questions |",
            "|-------|----------------|-------------|-------------|",
            f"| Branch 1 (n-hop) | {gv['branch1_nhop']['mean_precision']:.4f} | "
            f"{gv['branch1_nhop']['mean_recall']:.4f} | {gv['n_questions']} |",
            f"| Branch 2 (keyword) | {gv['branch2_keyword']['mean_precision']:.4f} | "
            f"{gv['branch2_keyword']['mean_recall']:.4f} | {gv['n_questions']} |",
            f"| Union | {gv['union']['mean_precision']:.4f} | "
            f"{gv['union']['mean_recall']:.4f} | {gv['n_questions']} |",
            f"| Kept after τ | {gv['kept_after_tau']['mean_precision']:.4f} | "
            f"{gv['kept_after_tau']['mean_recall']:.4f} | {gv['n_questions']} |",
            "",
        ]
    else:
        lines += [
            "## Graph-query validity at selected (n, τ)",
            "",
            block.get(
                "limitations",
                "Graph-query validity was not logged at the merged-selected config.",
            ),
            "",
        ]

    return lines


def per_pipeline_rows_at_merged_config(
    merged_block: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Extra per-pipeline table rows for SP-GQE / SP-GQE-i at merged (n*, tau*)."""
    sel = merged_block["selection"]
    n_star, tau_star = int(sel["n_hops"]), float(sel["tau"])
    rows: list[tuple[str, dict[str, Any]]] = []
    sp = merged_block["sp_gqe_at_selected_config"]
    rows.append(
        (
            f"SP-GQE(n={n_star},τ={tau_star})",
            {
                "mean_f1": sp["mean_f1"],
                "ci95_f1": sp["ci95_f1"],
                "mean_em": None,
                "mean_supporting_title_recall_at_k": None,
                "mean_retrieval_p_at_k": sp["mean_retrieval_p_at_k"],
                "n_seeds": sp["n_seeds"],
            },
        )
    )
    spi = merged_block.get("sp_gqe_i_at_selected_config")
    if spi:
        rows.append(
            (
                f"SP-GQE-i(n={n_star},τ={tau_star})",
                {
                    "mean_f1": spi["mean_f1"],
                    "ci95_f1": spi["ci95_f1"],
                    "mean_em": None,
                    "mean_supporting_title_recall_at_k": None,
                    "mean_retrieval_p_at_k": spi["mean_retrieval_p_at_k"],
                    "n_seeds": spi["n_seeds"],
                },
            )
        )
    else:
        rows.append(
            (
                f"SP-GQE-i(n={n_star},τ={tau_star})",
                {
                    "mean_f1": None,
                    "ci95_f1": None,
                    "mean_em": None,
                    "mean_supporting_title_recall_at_k": None,
                    "mean_retrieval_p_at_k": None,
                    "n_seeds": 0,
                    "pending_heatmap_sp_gqe_i": True,
                },
            )
        )
    return rows


def format_per_pipeline_row(label: str, a: dict[str, Any]) -> str:
    if a.get("pending_heatmap_sp_gqe_i"):
        return f"| {label} | — | — | — | — | — | 0 |"
    ci = a["ci95_f1"]
    mf1 = a["mean_f1"]
    mpk = a["mean_retrieval_p_at_k"]
    em = a["mean_em"]
    sup = a["mean_supporting_title_recall_at_k"]
    em_s = f"{em:.4f}" if em is not None else "—"
    sup_s = f"{sup:.4f}" if sup is not None else "—"
    f1_s = f"{mf1:.4f}" if mf1 is not None else "—"
    ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
    pk_s = f"{mpk:.4f}" if mpk is not None else "—"
    return (
        f"| {label} | {f1_s} | {ci_s} | {em_s} | {sup_s} | {pk_s} | {a['n_seeds']} |"
    )
