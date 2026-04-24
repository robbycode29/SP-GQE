# Aggregated results — 16 per-seed files

- **Total question instances:** 387
- **Seeds aggregated:** 2026-04-23:seed42(n=25), 2026-04-23:seed43(n=25), 2026-04-23:seed44(n=25), 2026-04-23:seed45(n=25), 2026-04-23:seed46(n=25), 2026-04-23:seed47(n=25), 2026-04-24:seed48(n=15), 2026-04-24:seed49(n=25), 2026-04-24:seed50(n=25), 2026-04-24:seed51(n=25), 2026-04-24:seed52(n=25), 2026-04-24:seed53(n=25), 2026-04-24:seed54(n=25), 2026-04-24:seed55(n=25), 2026-04-24:seed56(n=25), 2026-04-24:seed57(n=22)

## Per-pipeline (mean ± 95% CI across seed-level means)


| Pipeline            | Mean F1 | 95% CI F1        | Mean EM | Mean Sup-Title Recall@k | Mean P@k | n seeds |
| ------------------- | ------- | ---------------- | ------- | ----------------------- | -------- | ------- |
| V-RAG               | 0.5506  | [0.5062, 0.5949] | 0.4422  | 0.7793                  | 0.6735   | 16      |
| GQE-RAG(n=2)        | 0.5642  | [0.5167, 0.6118] | 0.4564  | 0.7905                  | 0.6605   | 16      |
| SP-GQE(n=2,τ=0.5)   | 0.5305  | [0.4930, 0.5680] | 0.4341  | 0.7687                  | 0.6378   | 16      |
| SP-GQE-i(n=3,τ=0.5) | 0.5448  | [0.5073, 0.5824] | 0.4556  | 0.7722                  | 0.6621   | 16      |
| GR-RAG              | 0.5560  | [0.5118, 0.6002] | 0.4472  | 0.7793                  | 0.6735   | 16      |
| GF-RAG              | 0.5478  | [0.5149, 0.5808] | 0.4429  | 0.7402                  | 0.6334   | 16      |


## Paired ΔF1 (SP-GQE(n=2, τ=0.5) − V-RAG) pooled across seeds


| Subset     | Mean Δ  | Bootstrap 95% CI  | n pairs |
| ---------- | ------- | ----------------- | ------- |
| bridge     | -0.0136 | [-0.0487, 0.0210] | 186     |
| comparison | -0.0247 | [-0.0648, 0.0149] | 201     |


## Graph-query validity (ablation, pooled per question)

*Supporting entities* are spaCy-NER entities extracted from the HotpotQA gold supporting paragraphs. Each row evaluates one stage of SP-GQE's graph side against that ground truth:

- **Branch 1 (SPARQL n-hop):** structural traversal from seed entities only.
- **Branch 2 (SPARQL keyword):** keyword-driven lookup over `rdfs:label` only.
- **Union:** the candidate pool that enters the τ pruner (before pruning).
- **Kept after τ=0.5:** the entities actually fed into the augmented FAISS query.


| Stage              | Mean precision | Mean recall | n questions |
| ------------------ | -------------- | ----------- | ----------- |
| Branch 1 (n-hop)   | 0.3352         | 0.5931      | 387         |
| Branch 2 (keyword) | 0.3813         | 0.1703      | 387         |
| Union              | 0.3072         | 0.6339      | 387         |
| Kept after τ=0.5   | 0.4479         | 0.2189      | 387         |


Interpretation: a rise in precision from Union → Kept indicates that the cosine-to-reunion pruner is removing noise; any drop in recall is the cost of that filtering. Branch 1 vs Branch 2 shows whether the two SPARQL queries are complementary (high union recall vs each branch alone) or redundant.