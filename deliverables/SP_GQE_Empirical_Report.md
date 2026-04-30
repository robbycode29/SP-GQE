# SP-GQE: Empirical evaluation on HotpotQA (distractor)

**Version:** 1.0 (generated from `results/AGGREGATED_`*, April 2026)  
**Repository:** `SP-GQE/`

---

## Abstract

We evaluate **SP-GQE** (Semantic Pruning on Graph-structured Query Expansion), a training-free pipeline that queries a per-question RDF co-occurrence graph with **SPARQL 1.1**, fuses two complementary branches, cosine-prunes entity candidates, and augments a dense (FAISS) retriever before a small language model reader. All pipelines share the same **corpus, embedding model, and reader**; only retrieval conditioning differs. On **562** stratified HotpotQA **distractor** development instances, aggregated across **23** random seeds, the primary two-branch **SP-GQE (n=2, τ=0.5)** configuration does **not** show a **statistically detectable** improvement in mean token F1 over **V-RAG**; paired 95% bootstrap confidence intervals for **(SP-GQE − V-RAG)** include zero on both **bridge** and **comparison** subsets. In-repo **GQE-RAG** and **GR-RAG** achieve slightly higher **mean** F1 in this aggregate. **Graph-query validity** diagnostics show complementary branch behaviour (union recall exceeds branch 2 alone) and a precision–recall trade-off after τ pruning, consistent with the design intent.

**Keywords:** multi-hop QA, HotpotQA, RAG, SPARQL, knowledge graph, semantic pruning, evaluation

---

## 1. Introduction and motivation

**Multi-hop open-domain question answering** requires a system to locate and combine information spread across several passages. **Retrieval-augmented generation (RAG)** (Lewis et al., 2020) remains a strong baseline: dense retrievers score paragraphs by similarity to a natural-language **question**, and a **reader** produces a short answer. When questions demand **linking** entities and relations, purely dense retrieval can miss crucial bridging facts unless the **query** explicitly reflects intermediate structure (Yang et al., 2018; Izacard & Grave, 2021, inter alia).

**Graph-augmented** pipelines attach **textual** or **relational** structure—community summaries, entity linking, re-ranking on graphs—across a range of designs (e.g. Edge et al., 2024, on “GraphRAG”-style document graphs; see also SubgraphRAG, HybGRAG, and related work). A recurring question is *where* structure should act (pre vs post retrieval) and whether **lightweight, query-time** graph expansion can improve answer quality under **fair** comparability (same retriever, same reader).

**SP-GQE** addresses a narrow slice of that space: a **per-question** co-occurrence **RDF** graph (built from HotpotQA’s **distractor** **10** paragraphs only—no external KB) is probed with **(i)** n-hop `spg:coOccur` **SPARQL** and **(ii)** `rdfs:label` keyword **SPARQL**; the union of entities is **cosine-pruned** (MiniLM) against a **reunion** of question + chunks; the kept entities **augment** the dense query string. The reader is a fixed **zero-temperature** Groq `llama-3.1-8b-instant` chat completion.

**Motivation for this report.** The dissertation pre-registered **H1**: improved answer **F1** vs **V-RAG** on **bridge**-type questions, with a **null** story on **comparison** types (`config/EXPERIMENT_PROTOCOL.md`). This document summarises the **cumulative** empirical run, reports **per-pipeline** and **paired** statistics, and documents **graph-query validity** ablations **requested by a reviewer** (precision/recall of entity sets vs gold supporting-paragraph NER).

**Scope and limitations.** We report **in-repository** baselines (V-RAG, GQE-RAG, SP-GQE, SP-GQE-i, GR-RAG, GF-RAG)—not a reproduction of every external **GraphRAG** leaderboard system. The sample is a **stratified subset** of the distractor **dev** split, not the full 7,405-question table used in some benchmark papers. Conclusions are **bounded** to this protocol, model, and sample size; absence of a significant difference does not prove parametric *equivalence* to V-RAG without a dedicated equivalence design.

---

## 2. Related work and citations


| Theme                                     | Reference (illustrative)                               | Notes for this work                                                                                            |
| ----------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Distractor multi-hop QA, explanation      | Yang et al. (2018) *HotpotQA*                          | 10-paragraph “distractor” setting; we use **dev** subset with stratified **bridge** / **comparison** sampling. |
| RAG for ODQA                              | Lewis et al. (2020) *Retrieval-augmented generation*   | **V-RAG** baseline encodes the same FAISS + reader stack.                                                      |
| Dense + graph hybrids                     | e.g. HybGRAG (SGA 2024 arXiv); SubgraphRAG (ICLR 2025) | Cited for positioning; not re-implemented.                                                                     |
| GraphRAG (community / graph of documents) | Edge et al. (2024) Microsoft GraphRAG (blog / arXiv)   | Different from our **in-corpus** co-occurrence **entity** graph; we cite as **situating** only.                |
| RAG vs GraphRAG systematics               | e.g. arXiv:2502.11371 (2025)                           | Motivates **fair** same-reader **comparability**.                                                              |


*Full reference list: see **References** below.*

---

## 3. Method summary (as implemented)

Implemented in `SP-GQE/`: per-question RDF (`RdfQuestionGraph`, `rdflib`); **FAISS** on `all-MiniLM-L6-v2` embeddings; **SPARQL** for Branch 1 (n-hop) and Branch 2 (label contains any keyword from noun-chunks); **τ=0.5** cosine pruning; **k** (default) top chunks to reader. Details: `config/EXPERIMENT_PROTOCOL.md`, `src/sp_gqe/experiment/`.

**Pipelines (same index + reader, graph path varies):**

- **V-RAG** — FAISS on the question only.  
- **GQE-RAG (n=2)** — branch 1 only, neighbours without τ pruning.  
- **SP-GQE (n=2, τ=0.5)** — both branches + pruning (primary).  
- **SP-GQE-i (n=3, τ=0.5)** — iterative single-branch variant with per-hop pruning.  
- **GR-RAG, GF-RAG** — **lexical** graph-adjacent controls (re-rank / filter entities).

**Metrics.** Primary: **token F1**; secondary: **EM**, **supporting-title recall @k**, **P@k**. **Paired** ΔF1: SP-GQE − V-RAG per (seed, qid), **bootstrap 95% CI** on the mean. **Graph-query validity:** P/R of entity sets vs NER on gold **supporting** spans at four **stages** (branch1, branch2, union, kept@τ). **Statistical design:** 95% CI on **seed-level** pipeline means (t on *n* seeds).

**Evaluation scale** (this report):

- **23** per-seed JSON files in `results/daily_runs/` (excluding `archive/`)  
- **562** **unique** (seed, qid) **question** instances in the aggregate  
- **Reader:** Groq `llama-3.1-8b-instant` (T=0), where runs used `--stack plan`

*Replication:* `python scripts/aggregate_daily_runs.py` reproduces `AGGREGATED_REPORT.md` / `AGGREGATED_SUMMARY.json`.

---

## 4. Results

### 4.1. Mean answer and retrieval metrics (aggregated)

**Table 1 — Per-pipeline means (95% CI across *seed*-level means; 23 seeds).** Source: `results/AGGREGATED_REPORT.md`.


| Pipeline               | Mean F1 | 95% CI (F1)      | Mean EM | Sup-title recall@k | P@k    |
| ---------------------- | ------- | ---------------- | ------- | ------------------ | ------ |
| GQE-RAG (n=2)          | 0.5760  | [0.5381, 0.6139] | 0.4636  | 0.8021             | 0.6654 |
| GR-RAG                 | 0.5727  | [0.5361, 0.6092] | 0.4607  | 0.7908             | 0.6738 |
| V-RAG                  | 0.5633  | [0.5276, 0.5991] | 0.4520  | 0.7908             | 0.6738 |
| SP-GQE-i (n=3,τ=0.5)   | 0.5520  | [0.5212, 0.5828] | 0.4560  | 0.7781             | 0.6610 |
| GF-RAG                 | 0.5512  | [0.5213, 0.5811] | 0.4403  | 0.7445             | 0.6382 |
| **SP-GQE (n=2,τ=0.5)** | 0.5489  | [0.5149, 0.5828] | 0.4463  | 0.7704             | 0.6360 |


**Takeaway:** **GQE-RAG** and **GR-RAG** sit at the top of **mean F1** in this aggregate; **SP-GQE** is **not** the strongest mean-F1 system here; **V-RAG** is **not** the weakest.

### 4.2. Paired contrast: SP-GQE vs V-RAG (token F1)

**Table 2 — Pooled paired ΔF1 = SP-GQE (n=2,τ=0.5) − V-RAG.**


| Subset     | Mean Δ  | Bootstrap 95% CI  | n pairs |
| ---------- | ------- | ----------------- | ------- |
| bridge     | −0.0185 | [−0.0473, 0.0093] | 270     |
| comparison | −0.0096 | [−0.0448, 0.0253] | 292     |


**Interpretation.** Both CIs **include 0**; the data **do not** support a **reliable** positive (or large negative) **average** gap in token F1 under this setup. H1 (bridge-specific win) is **not** **statistically** established at this **scale**; the **null** expectation on **comparison** is not contradicted in a “large win for SP-GQE” sense.

### 4.3. Graph-query validity (entity-level ablation, pooled, *n* = 562)

**Table 3 — Precision and recall of entity sets vs NER on gold supporting text.**


| Stage              | Mean P | Mean R | n questions |
| ------------------ | ------ | ------ | ----------- |
| Branch 1 (n-hop)   | 0.3460 | 0.5959 | 562         |
| Branch 2 (keyword) | 0.3793 | 0.1698 | 562         |
| Union (pre-τ)      | 0.3155 | 0.6343 | 562         |
| Kept (τ=0.5)       | 0.4545 | 0.2192 | 562         |


**Readout:** **Union** recovers a **higher** fraction of supporting entities than **branch 2** alone (recall 0.63 vs 0.17), so the two SPARQL branches are **complementary** at the entity-pool level. **τ** **increases** precision and **decreases** recall—consistent with **noise reduction** that may discard useful gold entities, possibly explaining part of the downstream F1 pattern.

### 4.4. Figures (from the latest `run_experiment` outputs)

Bar chart of mean F1 by pipeline (**last local run** may differ slightly from 23-seed **aggregate** above—use **tables** as canonical for the **full** study):

Mean F1 by pipeline (example run)

*Figure 1.* Bar chart: `results/pipelines_bar_f1.png`.

Single-seed **n×τ** heatmaps (produced when running **one** seed **without** `--no-heatmap`; for **sensitivity** of the SP-GQE design to **n** and **τ**):

SP-GQE mean F1 (n×τ) (example single-seed run)

*Figure 2.* `results/heatmap_fungi_n_tau.png` — **mean answer F1** for SP-GQE over a grid of hop depth **n** and threshold **τ** (illustrative).

SP-GQE mean P@k (n×τ) (example)

*Figure 3.* `results/heatmap_fungi_n_tau_retrieval_p_at_k.png` — mean retrieval P@*k* (same grid).

> **Note on figures 2–3:** The multi-seed aggregate in §4.1 uses a **fixed** (n=2, τ=0.5) configuration; the heatmaps **vary** n and τ on a **subsample** in a **single** heatmap run and are **illustrative** of hyperparameter sensitivity, not a second independent 23-seed aggregate.

---

## 5. Discussion

1. **Fair comparison.** The protocol holds **FAISS**, **embeddings**, and **reader** **fixed**; differences in Table 1 are plausibly attributed to **retrieval** **conditioning** and **entity** phrasing, not a different **LM** family.
2. **H1 and power.** A **null** in paired ΔF1 on **562** instances and **CIs** covering zero is **not** a proof of *equivalence*; it is evidence that, **here**, any **true** effect is **not** **large** relative to sampling noise. **Full** HotpotQA **dev** numbers would align better with “benchmark table” **norms** in the literature.
3. **“GraphRAG” claims.** The implemented **GR/GF** **controls** are **in-code**; external **graph-of-documents** systems (Edge et al., 2024) are **not** re-run. Claims should stay **in-family**.
4. **Graph validity** supports the **intended** mechanics (two **branches** + **pruner**); it does not **imply** that **higher** entity P/R on this proxy **translates** 1:1 to **F1**—the **reader** and **chunking** also dominate.
5. **Cost.** Groq TPD and `--no-heatmap` for most seeds were required for **practical** multi-seed **coverage** (`GROQ_MULTI_SEED_NOTE.md`).

---

## 6. Conclusion

In this **562-instance**, **23-seed** **aggregate**, **SP-GQE (n=2, τ=0.5)** does not exhibit a **clear, quantified** **improvement** in **token F1** over **V-RAG**; **paired** **intervals** **include** **zero** on both **bridge** and **comparison** **subsets**. **GQE-RAG** and **GR-RAG** **rank** **higher** in **mean** F1 in the same table. The **SPARQL** + **pruning** **ablation** on **entity** **sets** **is** **consistent** with a **complementary** **two-branch** **pool** and a **pruning** **trade-off**—a useful **mechanism** result even when **end-to-end** **answer** **F1** is **not** **superior** to the **classical** **RAG** **line**.

---

## References

- Edge, D. et al. (2024). *From local to global: a graph RAG approach to query-focused summarization.* Microsoft Research / GraphRAG (blog, technical report). *Used for high-level positioning only; not a reproduced baseline.*
- Izacard, G., & Grave, E. (2021). *Leveraging passage retrieval with generative models for open domain question answering.* In *EACL*.
- Lewis, P. et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* In *NeurIPS*.
- Yan, S. et al. (2022). *HybGRAG: Hybrid graph retrieval and reasoning for multi-hop questions.* (Representative **hybrid** RAG+graph work; arXiv / proceedings version as published.)
- Yang, Z. et al. (2018). *HotpotQA: A dataset for diverse, explainable multi-hop question answering.* In *EMNLP*.

*Optional survey (methodological framing):* arXiv:2502.11371 (2025). *A systematic comparison of RAG, GraphRAG, and other retrieval-augmented systems on text-based benchmarks* (title paraphrased; cite the exact arXiv entry you used in the dissertation bibliography).

*Note for thesis office:* Replace any **arXiv** lines with the **version** in your `.bib` file; add **SubgraphRAG (ICLR 2025)**, **HybGRAG**, and other arXiv IDs **as required** by your style guide (APA, IEEE, Chicago).

---

## Appendix: Seed inventory (aggregate input)

- **2026-04-23:** seeds 42–47, *n* = 25 each (6 files)  
- **2026-04-24:** seed 48 (*n* = 15); seeds 49–56 (*n* = 25); seed 57 (*n* = 22)  
- **2026-04-27:** seeds 58–64, *n* = 25 each (7 files)

**Total:** 562 question instances, 23 files.

*End of report.*