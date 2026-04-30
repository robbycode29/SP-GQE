#!/usr/bin/env python3
"""Visual debug: SP-GQE n×τ heatmap + GraphDB-style RDF subgraph (nodes + spg:coOccurs).

Reads a per-seed JSON from ``results/daily_runs/`` (must contain ``graph_query_log``
with ``cooccurrence_edges``, ``union_returned``, ``kept_tau``, ``seeds``).

Outputs (per ``--question-idx``):

* ``results/debug/kg_sidebyside_q{N}.png`` — left: RDF subgraph; right: heatmap
* ``results/debug/kg_heatmap_inset_q{N}.png`` — heatmap with graph inset (overlay)

Usage::

    python scripts/plot_kg_overlay.py \\
        --sample-json results/daily_runs/2026-04-23__seed42__n3.json \\
        --heatmap-json results/run_summary.json \\
        --question-idx 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO = Path(__file__).resolve().parents[1]


def _load_heatmap(json_path: Path) -> tuple[np.ndarray, list[int], list[float], str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = data.get("heatmap") or (data.get("per_seed", [{}])[0].get("heatmap", {}))
    if not h:
        raise SystemExit(f"No heatmap section found in {json_path}")
    return (
        np.array(h["mean_f1_grid"]),
        list(h["n_hops"]),
        list(h["tau"]),
        "SP-GQE mean F1 (n × τ)",
    )


def _node_role(
    node: str,
    seeds: set[str],
    kept: set[str],
) -> str:
    if node in seeds:
        return "seed"
    if node in kept:
        return "kept"
    return "pruned"


def _draw_graphdb_style(ax: plt.Axes, entry: dict) -> None:
    """Draw co-occurrence subgraph: Entity nodes, ``spg:coOccurs`` edges (like GraphDB)."""
    seeds = set(entry.get("seeds", []))
    kept = set(entry.get("kept_tau", []))
    edges_raw = entry.get("cooccurrence_edges") or []
    nodes_spotlight = set(entry.get("subgraph_spotlight_nodes", []))
    union = set(entry.get("union_returned", []))
    if not nodes_spotlight:
        nodes_spotlight = seeds | union

    g = nx.Graph()
    for n in nodes_spotlight:
        g.add_node(n)

    for pair in edges_raw:
        if len(pair) != 2:
            continue
        a, b = pair[0], pair[1]
        if a in g and b in g and a != b:
            g.add_edge(a, b, predicate="spg:coOccurs")

    # Isolated nodes still visible
    colors = []
    sizes = []
    for node in g.nodes:
        role = _node_role(node, seeds, kept)
        if role == "seed":
            colors.append("#c0392b")
            sizes.append(520)
        elif role == "kept":
            colors.append("#2980b9")
            sizes.append(380)
        else:
            colors.append("#95a5a6")
            sizes.append(260)

    if len(g) == 0:
        ax.text(0.5, 0.5, "(empty subgraph)", ha="center", va="center", fontsize=11)
        ax.axis("off")
        return

    pos = nx.spring_layout(g, seed=42, k=1.2 / max(1, np.sqrt(len(g))))
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        alpha=0.55,
        width=1.4,
        edge_color="#7f8c8d",
    )
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.92)
    labels = {n: (n if len(n) < 18 else n[:15] + "…") for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=6, ax=ax, font_weight="normal")

    ax.set_title(
        "RDF subgraph (per question)\n"
        "red = seed Entity · blue = kept after τ · gray = pruned · edge = spg:coOccurs",
        fontsize=9,
    )
    ax.axis("off")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)


def _draw_heatmap_panel(ax: plt.Axes, json_path: Path, title_suffix: str = "") -> None:
    grid, n_grid, tau_grid, title = _load_heatmap(json_path)
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(tau_grid)))
    ax.set_xticklabels([f"{t:.1f}" for t in tau_grid])
    ax.set_yticks(range(len(n_grid)))
    ax.set_yticklabels([str(n) for n in n_grid])
    ax.set_xlabel("τ (semantic prune)")
    ax.set_ylabel("n (SPARQL hop depth)")
    ax.set_title(title + title_suffix, fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean F1", fraction=0.046, pad=0.04)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-json", type=Path, required=True)
    ap.add_argument(
        "--heatmap-json",
        type=Path,
        default=REPO / "results" / "run_summary.json",
    )
    ap.add_argument("--question-idx", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results" / "debug",
        help="Directory for PNG outputs",
    )
    args = ap.parse_args()

    sample = json.loads(args.sample_json.read_text(encoding="utf-8"))
    log = sample.get("graph_query_log", [])
    if not log:
        raise SystemExit("graph_query_log is empty; re-run experiment first")
    if args.question_idx < 0 or args.question_idx >= len(log):
        raise SystemExit(f"question-idx must be in [0, {len(log) - 1}]")
    entry = log[args.question_idx]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qidx = args.question_idx

    # --- Side-by-side ---
    fig1, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    _draw_graphdb_style(axes[0], entry)
    try:
        _draw_heatmap_panel(axes[1], args.heatmap_json)
    except SystemExit as e:
        axes[1].text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)
        axes[1].axis("off")
    fig1.suptitle(f"Debug: {entry['question'][:100]}", fontsize=11, y=1.02)
    fig1.tight_layout()
    p1 = args.out_dir / f"kg_sidebyside_q{qidx}.png"
    fig1.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig1)
    print(f"Wrote {p1}")

    # --- Heatmap with graph inset (overlay) ---
    fig2, ax_h = plt.subplots(figsize=(8.2, 6.2))
    try:
        _draw_heatmap_panel(ax_h, args.heatmap_json, title_suffix=" + RDF subgraph inset")
        ax_inset = inset_axes(ax_h, width="38%", height="38%", loc="upper left", borderpad=2)
        _draw_graphdb_style(ax_inset, entry)
    except SystemExit as e:
        ax_h.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)
    fig2.suptitle(f"Debug overlay: {entry['question'][:90]}", fontsize=10, y=0.98)
    fig2.subplots_adjust(top=0.9, left=0.12, right=0.96, bottom=0.1)
    p2 = args.out_dir / f"kg_heatmap_inset_q{qidx}.png"
    fig2.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
