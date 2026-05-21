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

## Paired ΔF1 (SP-GQE(n=2, τ=0.5) − V-RAG) pooled across seeds

| Subset | Mean Δ | Bootstrap 95% CI | n pairs |
|--------|--------|------------------|---------|
| bridge | -0.0034 | [-0.0279, 0.0211] | 327 |
| comparison | -0.0086 | [-0.0375, 0.0202] | 353 |

## Graph-query validity (ablation, pooled per question)

*Supporting entities* are spaCy-NER entities extracted from the HotpotQA gold supporting paragraphs. Each row evaluates one stage of SP-GQE's graph side against that ground truth:

- **Branch 1 (SPARQL n-hop):** structural traversal from seed entities only.
- **Branch 2 (SPARQL keyword):** keyword-driven lookup over `rdfs:label` only.
- **Union:** the candidate pool that enters the τ pruner (before pruning).
- **Kept after τ=0.5:** the entities actually fed into the augmented FAISS query.

| Stage | Mean precision | Mean recall | n questions |
|-------|----------------|-------------|-------------|
| Branch 1 (n-hop) | 0.3411 | 0.5859 | 680 |
| Branch 2 (keyword) | 0.3737 | 0.1668 | 680 |
| Union | 0.3092 | 0.6256 | 680 |
| Kept after τ=0.5 | 0.4465 | 0.2146 | 680 |

Interpretation: a rise in precision from Union → Kept indicates that the cosine-to-reunion pruner is removing noise; any drop in recall is the cost of that filtering. Branch 1 vs Branch 2 shows whether the two SPARQL queries are complementary (high union recall vs each branch alone) or redundant.

## SP-GQE heatmap grid (merged n × τ)

*From* `results/merged_heatmap/merged_heatmap.json` *— sample-size-weighted mean over contributing seeds (total weight 580).*

| Metric | Best n_hops | Best τ | Merged mean |
|--------|-------------|--------|-------------|
| Mean answer F1 | 1 | 0.6 | 0.5766 |
| Mean retrieval P@k | 1 | 0.7 | 0.6486 |
