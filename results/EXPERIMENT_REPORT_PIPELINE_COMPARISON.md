# Pipeline comparison (merged-grid SP-GQE config)

SP-GQE / SP-GQE-i at **one** (n, τ) from the pooled heatmap F1 argmax. Per-seed values use that same cell. SP-GQE-i requires ``heatmap_sp_gqe_i`` in daily JSON (newer runs).

## Comparison table (mean token F1 and retrieval P@k)

| Pipeline | Mean F1 | 95% CI F1 | Mean P@k | 95% CI P@k | n seeds |
|----------|---------|-----------|----------|------------|---------|
| V-RAG | 0.5626 | [0.5313, 0.5940] | 0.6718 | [0.6540, 0.6896] | 28 |
| GQE-RAG(n=2) | 0.5622 | [0.5273, 0.5970] | 0.6630 | [0.6479, 0.6782] | 28 |
| GR-RAG | 0.5684 | [0.5363, 0.6004] | 0.6718 | [0.6540, 0.6896] | 28 |
| GF-RAG | 0.5465 | [0.5176, 0.5754] | 0.6308 | [0.6091, 0.6526] | 28 |
| SP-GQE (fixed n=2, τ=0.5) | 0.5557 | [0.5276, 0.5839] | 0.6344 | [0.6177, 0.6511] | 28 |
| SP-GQE-i (fixed n=3, τ=0.5) | 0.5443 | [0.5149, 0.5737] | 0.6615 | [0.6451, 0.6779] | 28 |
| SP-GQE(n=1,τ=0.6) | 0.5778 | [0.5501, 0.6055] | 0.6469 | [0.6469, 0.6469] | 24 |

## SP-GQE at merged-grid selected (n, τ)

**Selection:** argmax on merged sample-size-weighted mean_f1_grid. Global best on the pooled heatmap: **n=1, τ=0.6** (pooled mean F1 = 0.5766, P@k = 0.6486; weight = 580, 24 heatmap files).

Per-seed metrics use **that same (n, τ)** cell from each seed's stored heatmap (not per-seed argmax).

| Metric | Mean | 95% CI | n seeds |
|--------|------|--------|---------|
| Mean answer F1 | 0.5778 | [0.5501, 0.6055] | 24 |
| Mean retrieval P@k | 0.6469 | [0.6309, 0.6629] | 24 |

## Paired ΔF1 (SP-GQE − V-RAG)

Heatmap-selected config: **n=1, τ=0.6**. Question-level rows pool pairs from **heatmap seeds only**.

| Subset | Mean Δ | 95% CI | n |
|--------|--------|--------|---|
| bridge | -0.0114 | bootstrap [-0.0375, 0.0142] | 279 pairs |
| comparison | -0.0085 | bootstrap [-0.0398, 0.0222] | 301 pairs |
| all (seed-level; SP-GQE n=1, τ=0.6 vs V-RAG) | 0.0079 | t [-0.0133, 0.0292] | 24 seeds |

*Bridge/comparison pairs use SP-GQE from the experiment log (n=2, tau=0.5 (protocol default in daily_runs)), not SP-GQE(n=1, τ=0.6), because per-question F1 at the heatmap-selected cell was not stored. The **all (seed-level)** row uses the heatmap cell at (n*, τ*).*

## Graph-query validity at selected (n, τ)

Graph-query validity was not logged at the merged-selected config.

