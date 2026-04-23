---

## title: "Empirical Evaluation of Semantically-Pruned Graph Query Expansion (SP-GQE) for Multi-Hop Question Answering"

subtitle: "Consolidated findings, metrics, and positioning relative to retrieval baselines"
date: 2026
lang: en

## Abstract

Retrieval-augmented generation (RAG) remains a dominant paradigm for open-domain question answering, yet multi-hop questions often require linking evidence that dense vector similarity retrieves imperfectly. This chapter reports empirical results for **Semantically-Pruned Graph Query Expansion (SP-GQE)**, a training-free pipeline that expands a lightweight entity co-occurrence graph, prunes candidate entities by semantic similarity to sub-question probes, and augments the dense retriever’s query. Experiments use the **HotpotQA distractor development set** under a **pre-registered protocol** (`config/EXPERIMENT_PROTOCOL.md`), with **three independent stratified samples** (seeds 42, 43, 44; **20 questions per seed**, **60 instances** total), a shared **Groq-hosted Llama 3.1 8B Instant** reader, and **MiniLM** sentence embeddings. The **primary outcome** is mean **token-level F1**; secondary outcomes include **exact match (EM)** and **supporting-title recall @k**. Aggregated across seeds, **SP-GQE (n=2, τ=0.5)** achieves a **higher mean F1 than vanilla V-RAG** (0.530 vs 0.500), while **paired differences on bridge-type questions** (operationalizing multi-hop linking) are **positive on average** but **not statistically conclusive** at the current sample size (bootstrap 95% CI for mean ΔF1 crosses zero). The chapter situates these findings alongside representative RAG and GraphRAG references, discusses limitations of co-occurrence graphs and API budgets, and outlines directions for stronger graphs and larger-scale inference.

**Keywords:** retrieval-augmented generation; graph query expansion; HotpotQA; multi-hop question answering; semantic pruning.

---

## 1. Introduction

Multi-hop reading comprehension requires locating and combining facts scattered across multiple paragraphs. Dense retrievers excel at local semantic similarity but may miss entities whose connection is explicit only through **co-occurrence** or **discourse structure** within a fixed document set. Graph-based extensions to RAG—often grouped under “GraphRAG”—introduce relational or structural signals; however, unconstrained graph expansion can inject noise. **SP-GQE** addresses this tension by **expanding** a graph neighborhood and then **pruning** it using embedding similarity between candidate entities and automatically derived **sub-question probes**, before issuing a single augmented retrieval query.

This chapter **aggregates all empirical findings to date** from the SP-GQE reference implementation, with emphasis on **comparability**: every pipeline shares the **same chunk index**, **same embedding model**, and **same generative reader**, differing only in how the **retrieval query** is formed.

---

## 2. Background and related work

### 2.1 Retrieval-augmented generation

RAG couples a parametric reader with a non-parametric corpus index, typically via dense passage retrieval (Lewis et al., 2020). On HotpotQA-style multi-hop tasks, strong pipelines combine **high-recall retrieval** with **answer extraction or generation** (Yang et al., 2018).

### 2.2 Graph-augmented retrieval

Recent systems combine textual and graph signals: community-summarization GraphRAG (Edge et al., 2024), hybrid textual/relational retrieval (e.g. HybGRAG; ACL 2025), and subgraph-centric retrieval (SubgraphRAG; ICLR 2025). Surveys and systematic studies emphasize that **neither pure RAG nor pure GraphRAG dominates universally** (e.g. arXiv:2502.11371), motivating **query-aware** use of graph structure—consistent with SP-GQE’s pruning step.

**Positioning.** Table 1 summarizes **representative** systems (exact HotpotQA numbers vary by reader, year, and preprocessing; figures are **illustrative**, not reproduced experiments).

**Table 1. Representative systems discussed in related work (qualitative positioning).**


| System                                 | Reference                    | Notes on HotpotQA / setting                                       |
| -------------------------------------- | ---------------------------- | ----------------------------------------------------------------- |
| Vanilla RAG (dense retriever + reader) | Lewis et al., 2020           | Strong baselines; distractor setting is challenging.              |
| Microsoft GraphRAG                     | Edge et al., 2024            | Corpus-specific graphs; not identical preprocessing to this work. |
| HybGRAG                                | arXiv:2412.16311 (ACL 2025)  | Large gains on STaRK; HotpotQA not always primary.                |
| SubgraphRAG                            | arXiv:2410.20724 (ICLR 2025) | Trained subgraph scorer vs. our training-free pruning.            |
| RAG vs GraphRAG study                  | arXiv:2502.11371             | Motivates hybrid / conditional graph use.                         |


---

## 3. Research questions and hypotheses

The evaluation protocol is documented in `**config/EXPERIMENT_PROTOCOL.md`**. The pre-registered hypothesis is:

- **H1 (bridge / multi-hop):** SP-GQE **(n=2, τ=0.5)** improves **mean token F1** over **V-RAG** on **bridge** questions—questions that require **connecting two paragraphs**—holding reader and index fixed.
- **H2 (comparison, secondary):** On **comparison** questions, effects may be **smaller**, because two-entity comparison often remains tractable with strong dense retrieval alone.

**Operationalization.** HotpotQA labels `type ∈ {bridge, comparison}`. Stratified sampling draws **half bridge / half comparison** per seed (`sample_questions`).

---

## 4. Methodology

### 4.1 Dataset and sampling

- **Corpus:** HotpotQA **distractor** development split (`hotpot_dev_distractor_v1.json`).
- **Sampling:** For each RNG seed *s* ∈ {42, 43, 44}, **n = 20** questions: **10 bridge**, **10 comparison**, shuffled after stratification.
- **Total instances:** 3 × 20 = **60** (paired across pipelines).

### 4.2 Implementation stack (plan)


| Component          | Setting                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| Entity recognition | spaCy `en_core_web_sm`                                                        |
| Graph              | Sentence-level **entity co-occurrence**; **Neo4j** storage for the plan stack |
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalized)               |
| Vector index       | FAISS over sentence chunks `(title . sentence)`                               |
| Reader             | Groq API, `**llama-3.1-8b-instant`**, temperature 0, short-answer prompt      |
| Top-k retrieval    | *k* = 5                                                                       |


**Note:** The dissertation originally envisioned **Ollama** with **nomic-embed-text**; API quota and throughput motivated **Groq + MiniLM** for these runs. Comparisons **across pipelines within this chapter** remain valid; **absolute** comparison to external papers should be interpreted cautiously.

### 4.3 Pipelines (abbreviated)


| ID                        | Description                                                                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **V-RAG**                 | Question → dense retrieval → reader.                                                                                               |
| **GQE-RAG (n=2)**         | 2-hop entity neighborhood → augment query → retrieve → reader.                                                                     |
| **SP-GQE (n=2, τ=0.5)**   | 2-hop neighborhood → **semantic pruning** vs. sub-question probes (τ) → augment query (incl. verified probes) → retrieve → reader. |
| **SP-GQE-i (n=3, τ=0.5)** | **Iterative** 1-hop expansion with pruning each hop (3 hops).                                                                      |
| **GR-RAG**                | Vanilla retrieval, then **re-rank** chunks by entity overlap with question.                                                        |
| **GF-RAG**                | **Filter** chunks by question-entity hit, then retrieve within filtered pool.                                                      |


Full procedural detail appears in `src/sp_gqe/experiment/pipelines.py`.

### 4.4 Metrics


| Role       | Metric                         | Definition                                                                                                  |
| ---------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Primary    | **Token F1**                   | Normalized token overlap between prediction and gold short answer.                                          |
| Secondary  | **Exact match (EM)**           | 1 if normalized strings match, else 0; averaged per pipeline.                                               |
| Secondary  | **Supporting-title recall @k** | Fraction of **gold supporting paragraph titles** covered by at least one of the **top-k** retrieved chunks. |
| Diagnostic | **Retrieval P@k**              | Fraction of top-k chunks whose **title** matches a supporting fact (related but not identical to recall).   |


### 4.5 Statistical analysis

- **Across seeds:** For each pipeline, compute the **mean F1 per seed**, then report the **mean of those seed-level means** with a **95% confidence interval** using the **t-distribution** with *k*−1 degrees of freedom (*k* = 3 seeds). **EM** and **supporting-title recall** CIs are **clipped to [0, 1]**.
- **Mechanism test:** For each question, compute **ΔF1 = F1(SP-GQE) − F1(V-RAG)**. Pool all **bridge** pairs (30 total) and all **comparison** pairs (30). Report **mean ΔF1** and a **bootstrap 95% CI** (8,000 resamples) for the mean difference.

**Limitation:** With *k* = 3 seeds, intervals on **seed-level means** are **wide**; the design prioritizes **diversity of draws** over **tight confidence bounds**.

### 4.6 Computational note

The final multi-seed run used `**--no-heatmap`** to omit the **n × τ** ablation grid (15 extra LLM calls per question), staying within **daily Groq token** limits. Earlier **single-seed** runs with full heatmaps exist in project history; they are **not** the primary aggregate reported here.

---

## 5. Results

### 5.1 Aggregate performance across three seeds

**Table 2. Mean metrics aggregated across seeds (95% CI on seed-level means; *k* = 3).**


| Pipeline                | Mean F1   | 95% CI             | Mean EM   | 95% CI             | Mean sup. title recall@k | 95% CI         |
| ----------------------- | --------- | ------------------ | --------- | ------------------ | ------------------------ | -------------- |
| V-RAG                   | 0.500     | [0.238, 0.761]     | 0.417     | [0.158, 0.675]     | 0.842                    | [0.747, 0.937] |
| GQE-RAG (n=2)           | 0.510     | [0.233, 0.787]     | 0.433     | [0.175, 0.692]     | 0.800                    | [0.585, 1.000] |
| **SP-GQE (n=2, τ=0.5)** | **0.530** | **[0.348, 0.713]** | **0.500** | **[0.376, 0.624]** | 0.817                    | [0.722, 0.912] |
| SP-GQE-i (n=3, τ=0.5)   | 0.491     | [0.253, 0.728]     | 0.433     | [0.244, 0.623]     | 0.800                    | [0.738, 0.862] |
| GR-RAG                  | 0.535     | [0.389, 0.682]     | 0.450     | [0.326, 0.574]     | 0.842                    | [0.747, 0.937] |
| GF-RAG                  | 0.547     | [0.297, 0.797]     | 0.483     | [0.225, 0.742]     | 0.783                    | [0.747, 0.819] |


**Observation.** **GF-RAG** and **GR-RAG** achieve the **highest mean F1** in this aggregate table; **SP-GQE** ranks **third** by mean F1 but **first** among **mean EM**. **V-RAG** attains the **highest supporting-title recall** jointly with GR-RAG, illustrating **decoupling** between title-overlap retrieval metrics and **answer F1**.

### 5.2 Seed-level variability (primary metric)

**Table 3. Mean token F1 by seed (SP-GQE vs V-RAG).**


| Seed | V-RAG | SP-GQE (n=2, τ=0.5) |
| ---- | ----- | ------------------- |
| 42   | 0.541 | 0.547               |
| 43   | 0.380 | 0.450               |
| 44   | 0.578 | 0.594               |


Seed **43** yields **lower** absolute F1 for both systems, driving **wide** aggregate CIs.

### 5.3 Paired comparison: SP-GQE vs V-RAG

**Table 4. Paired mean difference in token F1 (SP-GQE − V-RAG).**


| Stratum         | n pairs | Mean ΔF1 | Bootstrap 95% CI |
| --------------- | ------- | -------- | ---------------- |
| Bridge (H1)     | 30      | +0.052   | [−0.075, 0.185]  |
| Comparison (H2) | 30      | +0.009   | [−0.088, 0.111]  |


The **bridge** subset shows a **positive point estimate** consistent with H1, but the **confidence interval includes zero**; H1 is **not confirmed** at conventional levels for this sample. The **comparison** estimate is **near zero**.

### 5.4 Figure

**Figure 1.** Mean token F1 by pipeline (aggregated across three seeds; same 60 question instances per pipeline).

Mean token F1 by pipeline

*Source: `results/run_summary.json`; figure regenerated via `sp_gqe.experiment.plots.bar_comparison`.*

### 5.5 Supplementary figures (single-seed *n* × τ ablation)

The following heatmaps were produced by an **earlier single-seed** run (same codebase; **not** part of the three-seed aggregate in Table 2). They illustrate **sensitivity** of mean F1 to pruning threshold **τ** and hop count **n** for the SP-GQE pipeline variant.

**Figure 2.** Mean answer F1 over a 3 × 5 grid of (*n* hops, τ) — exploratory.

SP-GQE mean F1 heatmap n by tau

**Figure 3.** Mean retrieval P@k (title overlap with supporting facts) over the same grid.

SP-GQE retrieval P@k heatmap

---

## 6. Discussion

**Relative performance.** Under identical code paths, **graph-augmented** and **graph-filtered** variants (GQE-RAG, SP-GQE, GR-RAG, GF-RAG) **do not uniformly dominate** V-RAG; **GF-RAG** and **GR-RAG** achieve the **highest mean F1** here, while **SP-GQE** improves **EM** and shows a **modest mean F1 gain** over V-RAG. This aligns with literature suggesting **conditional** benefits of graph structure (arXiv:2502.11371).

**Mechanism vs metric.** Higher **supporting-title recall** does not imply higher **answer F1** (e.g. GF-RAG): readers can be **misled by chunk order** or **partially relevant** passages.

**Iterative SP-GQE-i.** With **(n=3, τ=0.5)**, mean F1 **falls below** batch SP-GQE in the aggregate table—consistent with **over-pruning** or **query phrasing** differences (iterative variant omits the explicit “verified probes” string present in batch SP-GQE).

**External comparability.** Published HotpotQA F1 scores (e.g. high-50s to low-60s for strong pipelines) assume **different readers, retrievers, and often full dev** evaluation. This chapter emphasizes **internal** ranking and **hypothesis-oriented** contrasts.

---

## 7. Limitations and future work

1. **Sample size:** 20 questions × 3 seeds limits power; **more seeds** and/or **larger *n*** tighten intervals.
2. **Graph semantics:** Co-occurrence is **shallow**; **OpenIE**, **dependency**, or **Wikidata-linked** edges may better expose multi-hop structure—**without** abandoning HotpotQA if answers remain **passage-grounded**.
3. **API / model drift:** Groq limits and model **versioning** constrain reproducibility; **frozen model IDs** and **logged prompts** are essential.
4. **Heatmap ablation:** The **n × τ** grid was omitted in the final aggregate run to save tokens; a **full grid** on a larger subset is needed to **select** hyperparameters without circularity.

---

## 8. Conclusion

This chapter consolidates empirical evidence for **SP-GQE** on HotpotQA distractor dev under a **transparent protocol**. **SP-GQE (n=2, τ=0.5)** achieves **higher mean token F1 and EM than V-RAG** in the **three-seed aggregate**, and **bridge-question paired differences** favor SP-GQE **on average**, but **uncertainty remains large** and **H1 is not statistically established** at the current scale. The results support **continued investigation** of **semantic pruning** with **richer graphs** and **larger evaluation budgets**, while treating **graph filtering and re-ranking** baselines (GF-RAG, GR-RAG) as **strong competitors** in the same codebase.

---

## References

Edge, D., et al. (2024). *From local to global: A graph RAG approach to query-focused summarization.* Microsoft Research / GraphRAG materials.

Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.

Yang, Z., et al. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. *International Conference on Machine Learning (ICML)*.

HybGRAG (2025). Hybrid retrieval over textual and relational signals. *ACL 2025* (see arXiv:2412.16311).

SubgraphRAG (2025). Subgraph retrieval for knowledge-intensive QA. *ICLR 2025* (see arXiv:2410.20724).

Anonymous / preprint (2025). Systematic comparison of RAG and GraphRAG families on text benchmarks. arXiv:2502.11371.

---

## Appendix A. Data and artifact paths


| Artifact                           | Path (relative to repository root `SP-GQE/`)                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------- |
| Protocol                           | `config/EXPERIMENT_PROTOCOL.md`                                                       |
| Machine-readable results           | `results/run_summary.json`                                                            |
| Short experiment report            | `results/EXPERIMENT_REPORT.md`                                                        |
| Bar chart                          | `results/pipelines_bar_f1.png`                                                        |
| Heatmaps (exploratory single-seed) | `results/heatmap_fungi_n_tau.png`, `results/heatmap_fungi_n_tau_retrieval_p_at_k.png` |
| Groq run notes                     | `results/GROQ_MULTI_SEED_NOTE.md`                                                     |


---

## Appendix B. Conversion to Microsoft Word

From the `SP-GQE` directory, **Pandoc** typically preserves headings, tables, and figures:

```bash
pandoc dissertation/EMPIRICAL_CHAPTER.md -o dissertation/EMPIRICAL_CHAPTER.docx --resource-path=.
```

If images fail to embed, ensure `--resource-path` includes the repository root or use absolute paths.