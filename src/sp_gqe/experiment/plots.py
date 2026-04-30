"""Visualization: n × τ heatmap with organic / growth-inspired colormap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.ndimage import gaussian_filter


def fungi_heatmap(
    grid: np.ndarray,
    n_vals: list[int],
    tau_vals: list[float],
    out_path: Path,
    title: str = "SP-GQE: mean F1 over held questions (n hops × τ)",
    cbar_label: str = "Mean answer F1",
) -> None:
    """
    Smooth slightly for an organic, continuous field look (mycelium / growth metaphor).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(grid, dtype=np.float64)
    z = gaussian_filter(z, sigma=0.35)

    fig, ax = plt.subplots(figsize=(9, 6))
    # Earthy / growth palette: dark soil → moss → pale hyphae
    cmap = colors.LinearSegmentedColormap.from_list(
        "mycelium",
        ["#1a0f0a", "#2d5016", "#5a9216", "#a3d95c", "#e8f5d4"],
        N=256,
    )
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
        extent=(
            tau_vals[0] - 0.05,
            tau_vals[-1] + 0.05,
            n_vals[0] - 0.5,
            n_vals[-1] + 0.5,
        ),
    )
    ax.set_xlabel("Semantic pruning threshold τ (tighter →)")
    ax.set_ylabel("Graph expansion depth n (hops)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(tau_vals)
    ax.set_xticklabels([f"{t:.2f}" for t in tau_vals])
    ax.set_yticks(n_vals)
    ax.set_yticklabels([str(n) for n in n_vals])

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def bar_comparison(
    labels: list[str],
    values: list[float],
    out_path: Path,
    title: str = "Pipeline mean F1 (same subset)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#3d6b2c", edgecolor="#1a3014")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Mean F1")
    ax.set_title(title)
    ax.set_ylim(0, max(0.15, max(values) * 1.15) if values else 1.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
