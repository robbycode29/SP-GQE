# Experiment run (`--stack plan`)

- **Protocol:** `config\EXPERIMENT_PROTOCOL.md`
- **Hypothesis:** SP-GQE(n=2,τ=0.5) improves mean token F1 vs V-RAG on bridge (multi-hop) questions; comparison subset is secondary.
- **Primary metric:** mean_token_f1; **secondary:** answer_exact_match, supporting_title_recall_at_k

- **Seeds:** [58, 59, 60, 61] (4 runs × 25 questions = 100 instances).
- **Stack (plan):** Groq API `llama-3.1-8b-instant` generation (T=0), `all-MiniLM-L6-v2` embeddings, RDFLib in-memory per-question RDF graph queried via SPARQL 1.1, FAISS.

## Aggregated across seeds (mean ± 95% CI on seed-level means)

| Pipeline | Mean F1 | 95% CI | Mean EM | 95% CI | Mean sup. title recall@k | 95% CI |
|----------|---------|--------|---------|--------|---------------------------|--------|
| V-RAG | 0.6142 | [0.4853, 0.7431] | 0.4800 | [0.2535, 0.7065] | 0.8350 | [0.7950, 0.8750] |
| GQE-RAG(n=2) | 0.6310 | [0.4692, 0.7928] | 0.5000 | [0.2535, 0.7465] | 0.8350 | [0.7598, 0.9102] |
| SP-GQE(n=2,τ=0.5) | 0.6296 | [0.4843, 0.7749] | 0.5000 | [0.2826, 0.7174] | 0.7900 | [0.7582, 0.8218] |
| SP-GQE-i(n=3,τ=0.5) | 0.6174 | [0.5354, 0.6993] | 0.5100 | [0.3788, 0.6412] | 0.8200 | [0.7619, 0.8781] |
| GR-RAG | 0.6351 | [0.4879, 0.7822] | 0.5200 | [0.2935, 0.7465] | 0.8350 | [0.7950, 0.8750] |
| GF-RAG | 0.5678 | [0.3839, 0.7516] | 0.4400 | [0.1908, 0.6892] | 0.7600 | [0.7150, 0.8050] |

## Mechanism test (paired SP-GQE − V-RAG on token F1)

- **Bridge (H1):** mean Δ = -0.0252, bootstrap 95% CI [-0.0891, 0.0293], n = 48
- **Comparison:** mean Δ = 0.0528, bootstrap 95% CI [-0.0184, 0.1321], n = 52

*Heatmaps skipped (`--no-heatmap` or multi-seed).*

## Positioning vs published RAG / GraphRAG systems

| System | Reference | Notes |
|--------|-----------|-------|
| Vanilla RAG (DPR-style dense retriever + reader) | Lewis et al., NeurIPS 2020; follow-up retrieval stacks | Depends on reader; HotpotQA distractor is harder than full-wiki. |
| Microsoft GraphRAG (community summaries) | Edge et al., 2024; msft graphrag | Heavy offline indexing; not directly comparable sample-for-sample. |
| HybGRAG | arXiv:2412.16311 (ACL 2025) | Demonstrates hybrid retrieval > single-modality. |
| SubgraphRAG | ICLR 2025; arXiv:2410.20724 | Trained scorer vs our training-free pruning. |
| RAG vs GraphRAG systematic study | arXiv:2502.11371 (Feb 2025) | Motivates hybrid / query-aware graph use (aligned with SP-GQE). |
| SP-GQE (this work) | Repository experiments | Prototype stack: meant for ablations vs baselines under identical code. |
