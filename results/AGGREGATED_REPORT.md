# Aggregated results — 28 per-seed files

- **Total question instances:** 680
- **Seeds aggregated:** 2026-05-07:seed42(n=25), 2026-05-08:seed43(n=25), 2026-05-08:seed44(n=25), 2026-05-08:seed45(n=25), 2026-05-11:seed46(n=25), 2026-05-11:seed47(n=25), 2026-05-11:seed48(n=15), 2026-05-12:seed49(n=25), 2026-05-12:seed50(n=25), 2026-05-12:seed51(n=25), 2026-05-13:seed52(n=25), 2026-05-13:seed53(n=25), 2026-05-14:seed54(n=25), 2026-05-14:seed55(n=25), 2026-05-18:seed56(n=25), 2026-05-18:seed57(n=22), 2026-05-18:seed58(n=25), 2026-04-27:seed59(n=25), 2026-05-19:seed60(n=25), 2026-05-20:seed61(n=25), 2026-05-20:seed62(n=25), 2026-05-21:seed63(n=25), 2026-05-21:seed64(n=25), 2026-05-21:seed65(n=18), 2026-05-06:seed66(n=25), 2026-05-07:seed67(n=25), 2026-05-07:seed68(n=25), 2026-05-07:seed69(n=25)

## Per-pipeline (mean ± 95% CI across seed-level means)

| Pipeline | Mean F1 | 95% CI F1 | Mean EM | Mean Sup-Title Recall@k | Mean P@k | n seeds |
|----------|---------|-----------|---------|-------------------------|----------|---------|
| V-RAG | 0.5626 | [0.5313, 0.5940] | 0.4521 | 0.7842 | 0.6718 | 28 |
| GQE-RAG(n=2) | 0.5622 | [0.5273, 0.5970] | 0.4576 | 0.7913 | 0.6630 | 28 |
| SP-GQE(n=2,τ=0.5) | 0.5557 | [0.5276, 0.5839] | 0.4507 | 0.7634 | 0.6344 | 28 |
| SP-GQE-i(n=3,τ=0.5) | 0.5443 | [0.5149, 0.5737] | 0.4469 | 0.7781 | 0.6615 | 28 |
| GR-RAG | 0.5684 | [0.5363, 0.6004] | 0.4626 | 0.7842 | 0.6718 | 28 |
| GF-RAG | 0.5465 | [0.5176, 0.5754] | 0.4352 | 0.7321 | 0.6308 | 28 |

*Merged-grid selected (n, τ) for SP-GQE / SP-GQE-i below: **n=1, τ=0.6** (argmax on pooled heatmap F1). EM / sup-title recall not swept in heatmap runs (—). SP-GQE-i row needs ``heatmap_sp_gqe_i`` in daily JSON (added in newer ``run_experiment.py`` runs).*

| SP-GQE(n=1,τ=0.6) | 0.5778 | [0.5501, 0.6055] | — | — | 0.6469 | 24 |
| SP-GQE-i(n=1,τ=0.6) | — | — | — | — | — | 0 |

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


## Appendix: protocol default (n=2, τ=0.5)

Logged during experiments before merged-grid selection was applied to reporting. Use merged-config sections above as primary when available.

### Paired ΔF1 (SP-GQE(n=2, τ=0.5) − V-RAG), question-level

| Subset | Mean Δ | Bootstrap 95% CI | n pairs |
|--------|--------|------------------|---------|
| bridge | -0.0034 | [-0.0279, 0.0211] | 327 |
| comparison | -0.0086 | [-0.0375, 0.0202] | 353 |

### Graph-query validity (n=2, τ=0.5 log)

| Stage | Mean precision | Mean recall | n questions |
|-------|----------------|-------------|-------------|
| Branch 1 (n-hop) | 0.3411 | 0.5859 | 680 |
| Branch 2 (keyword) | 0.3737 | 0.1668 | 680 |
| Union | 0.3092 | 0.6256 | 680 |
| Kept after τ=0.5 | 0.4465 | 0.2146 | 680 |
