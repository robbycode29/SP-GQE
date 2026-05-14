# Aggregated results — 28 per-seed files

- **Total question instances:** 680
- **Seeds aggregated:** 2026-05-07:seed42(n=25), 2026-05-08:seed43(n=25), 2026-05-08:seed44(n=25), 2026-05-08:seed45(n=25), 2026-05-11:seed46(n=25), 2026-05-11:seed47(n=25), 2026-05-11:seed48(n=15), 2026-05-12:seed49(n=25), 2026-05-12:seed50(n=25), 2026-05-12:seed51(n=25), 2026-04-24:seed52(n=25), 2026-04-24:seed53(n=25), 2026-04-24:seed54(n=25), 2026-04-24:seed55(n=25), 2026-04-24:seed56(n=25), 2026-04-24:seed57(n=22), 2026-04-27:seed58(n=25), 2026-04-27:seed59(n=25), 2026-04-27:seed60(n=25), 2026-04-27:seed61(n=25), 2026-04-27:seed62(n=25), 2026-04-27:seed63(n=25), 2026-04-27:seed64(n=25), 2026-05-06:seed65(n=18), 2026-05-06:seed66(n=25), 2026-05-07:seed67(n=25), 2026-05-07:seed68(n=25), 2026-05-07:seed69(n=25)

## Per-pipeline (mean ± 95% CI across seed-level means)

| Pipeline | Mean F1 | 95% CI F1 | Mean EM | Mean Sup-Title Recall@k | Mean P@k | n seeds |
|----------|---------|-----------|---------|-------------------------|----------|---------|
| V-RAG | 0.5618 | [0.5303, 0.5932] | 0.4506 | 0.7842 | 0.6718 | 28 |
| GQE-RAG(n=2) | 0.5555 | [0.5216, 0.5893] | 0.4458 | 0.7898 | 0.6591 | 28 |
| SP-GQE(n=2,τ=0.5) | 0.5537 | [0.5235, 0.5839] | 0.4470 | 0.7634 | 0.6320 | 28 |
| SP-GQE-i(n=3,τ=0.5) | 0.5443 | [0.5138, 0.5749] | 0.4469 | 0.7774 | 0.6621 | 28 |
| GR-RAG | 0.5672 | [0.5351, 0.5993] | 0.4578 | 0.7842 | 0.6718 | 28 |
| GF-RAG | 0.5452 | [0.5167, 0.5736] | 0.4324 | 0.7321 | 0.6308 | 28 |

## Paired ΔF1 (SP-GQE(n=2, τ=0.5) − V-RAG) pooled across seeds

| Subset | Mean Δ | Bootstrap 95% CI | n pairs |
|--------|--------|------------------|---------|
| bridge | -0.0132 | [-0.0388, 0.0119] | 327 |
| comparison | -0.0024 | [-0.0337, 0.0285] | 353 |

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

*From* `results/merged_heatmap/merged_heatmap.json` *— sample-size-weighted mean over contributing seeds (total weight 283).*

| Metric | Best n_hops | Best τ | Merged mean |
|--------|-------------|--------|-------------|
| Mean answer F1 | 3 | 0.6 | 0.5693 |
| Mean retrieval P@k | 1 | 0.6 | 0.6473 |
