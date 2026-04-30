# Aggregated results — 23 per-seed files

- **Total question instances:** 562
- **Seeds aggregated:** 2026-04-23:seed42(n=25), 2026-04-23:seed43(n=25), 2026-04-23:seed44(n=25), 2026-04-23:seed45(n=25), 2026-04-23:seed46(n=25), 2026-04-23:seed47(n=25), 2026-04-24:seed48(n=15), 2026-04-24:seed49(n=25), 2026-04-24:seed50(n=25), 2026-04-24:seed51(n=25), 2026-04-24:seed52(n=25), 2026-04-24:seed53(n=25), 2026-04-24:seed54(n=25), 2026-04-24:seed55(n=25), 2026-04-24:seed56(n=25), 2026-04-24:seed57(n=22), 2026-04-27:seed58(n=25), 2026-04-27:seed59(n=25), 2026-04-27:seed60(n=25), 2026-04-27:seed61(n=25), 2026-04-27:seed62(n=25), 2026-04-27:seed63(n=25), 2026-04-27:seed64(n=25)

## Per-pipeline (mean ± 95% CI across seed-level means)


| Pipeline            | Mean F1 | 95% CI F1        | Mean EM | Mean Sup-Title Recall@k | Mean P@k | n seeds |
| ------------------- | ------- | ---------------- | ------- | ----------------------- | -------- | ------- |
| V-RAG               | 0.5633  | [0.5276, 0.5991] | 0.4520  | 0.7908                  | 0.6738   | 23      |
| GQE-RAG(n=2)        | 0.5760  | [0.5381, 0.6139] | 0.4636  | 0.8021                  | 0.6654   | 23      |
| SP-GQE(n=2,τ=0.5)   | 0.5489  | [0.5149, 0.5828] | 0.4463  | 0.7704                  | 0.6360   | 23      |
| SP-GQE-i(n=3,τ=0.5) | 0.5520  | [0.5212, 0.5828] | 0.4560  | 0.7781                  | 0.6610   | 23      |
| GR-RAG              | 0.5727  | [0.5361, 0.6092] | 0.4607  | 0.7908                  | 0.6738   | 23      |
| GF-RAG              | 0.5512  | [0.5213, 0.5811] | 0.4403  | 0.7445                  | 0.6382   | 23      |


## Paired ΔF1 (SP-GQE(n=2, τ=0.5) − V-RAG) pooled across seeds


| Subset     | Mean Δ  | Bootstrap 95% CI  | n pairs |
| ---------- | ------- | ----------------- | ------- |
| bridge     | -0.0185 | [-0.0473, 0.0093] | 270     |
| comparison | -0.0096 | [-0.0448, 0.0253] | 292     |


## Graph-query validity (ablation, pooled per question)

*Supporting entities* are spaCy-NER entities extracted from the HotpotQA gold supporting paragraphs. Each row evaluates one stage of SP-GQE's graph side against that ground truth:

- **Branch 1 (SPARQL n-hop):** structural traversal from seed entities only.
- **Branch 2 (SPARQL keyword):** keyword-driven lookup over `rdfs:label` only.
- **Union:** the candidate pool that enters the τ pruner (before pruning).
- **Kept after τ=0.5:** the entities actually fed into the augmented FAISS query.


| Stage              | Mean precision | Mean recall | n questions |
| ------------------ | -------------- | ----------- | ----------- |
| Branch 1 (n-hop)   | 0.3460         | 0.5959      | 562         |
| Branch 2 (keyword) | 0.3793         | 0.1698      | 562         |
| Union              | 0.3155         | 0.6343      | 562         |
| Kept after τ=0.5   | 0.4545         | 0.2192      | 562         |


Interpretation: a rise in precision from Union → Kept indicates that the cosine-to-reunion pruner is removing noise; any drop in recall is the cost of that filtering. Branch 1 vs Branch 2 shows whether the two SPARQL queries are complementary (high union recall vs each branch alone) or redundant.