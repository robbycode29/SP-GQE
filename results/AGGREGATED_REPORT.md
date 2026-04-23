# Aggregated results — 6 per-seed files

- **Total question instances:** 150
- **Seeds aggregated:** 2026-04-23:seed42(n=25), 2026-04-23:seed43(n=25), 2026-04-23:seed44(n=25), 2026-04-23:seed45(n=25), 2026-04-23:seed46(n=25), 2026-04-23:seed47(n=25)

## Per-pipeline (mean ± 95% CI across seed-level means)

| Pipeline | Mean F1 | 95% CI F1 | Mean EM | Mean Sup-Title Recall@k | Mean P@k | n seeds |
|----------|---------|-----------|---------|-------------------------|----------|---------|
| V-RAG | 0.5419 | [0.4262, 0.6575] | 0.4467 | 0.7967 | 0.7053 | 6 |
| GQE-RAG(n=2) | 0.5727 | [0.4808, 0.6645] | 0.4733 | 0.7900 | 0.6760 | 6 |
| SP-GQE(n=2,τ=0.5) | 0.5401 | [0.4645, 0.6156] | 0.4533 | 0.7633 | 0.6640 | 6 |
| SP-GQE-i(n=3,τ=0.5) | 0.5374 | [0.4762, 0.5986] | 0.4533 | 0.7800 | 0.6960 | 6 |
| GR-RAG | 0.5652 | [0.4484, 0.6820] | 0.4667 | 0.7967 | 0.7053 | 6 |
| GF-RAG | 0.5451 | [0.4565, 0.6336] | 0.4600 | 0.7533 | 0.6573 | 6 |

## Paired ΔF1 (SP-GQE(n=2, τ=0.5) − V-RAG) pooled across seeds

| Subset | Mean Δ | Bootstrap 95% CI | n pairs |
|--------|--------|------------------|---------|
| bridge | 0.0160 | [-0.0404, 0.0756] | 72 |
| comparison | -0.0182 | [-0.0877, 0.0474] | 78 |

## Graph-query validity (ablation, pooled per question)

*Supporting entities* are spaCy-NER entities extracted from the HotpotQA gold supporting paragraphs. Each row evaluates one stage of SP-GQE's graph side against that ground truth:

- **Branch 1 (SPARQL n-hop):** structural traversal from seed entities only.
- **Branch 2 (SPARQL keyword):** keyword-driven lookup over `rdfs:label` only.
- **Union:** the candidate pool that enters the τ pruner (before pruning).
- **Kept after τ=0.5:** the entities actually fed into the augmented FAISS query.

| Stage | Mean precision | Mean recall | n questions |
|-------|----------------|-------------|-------------|
| Branch 1 (n-hop) | 0.3507 | 0.6056 | 150 |
| Branch 2 (keyword) | 0.3981 | 0.1799 | 150 |
| Union | 0.3121 | 0.6462 | 150 |
| Kept after τ=0.5 | 0.4545 | 0.2143 | 150 |

Interpretation: a rise in precision from Union → Kept indicates that the cosine-to-reunion pruner is removing noise; any drop in recall is the cost of that filtering. Branch 1 vs Branch 2 shows whether the two SPARQL queries are complementary (high union recall vs each branch alone) or redundant.
