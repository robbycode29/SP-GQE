# Experiment run (`--stack plan`)

- **Sample:** 20 HotpotQA distractor dev questions (seed 42).
- **Stack (plan):** Groq API `llama-3.1-8b-instant` generation (T=0), `all-MiniLM-L6-v2` embeddings, Neo4j Bolt co-occurrence graph per question, FAISS.
- **KG:** spaCy `en_core_web_sm` entities; **co-occurrence** edges (sentence-level). Dissertation LLM triple extraction can replace edge construction later.

## Mean answer F1


| Pipeline            | Mean F1 | Std    |
| ------------------- | ------- | ------ |
| V-RAG               | 0.5412  | 0.4602 |
| GQE-RAG(n=2)        | 0.5160  | 0.4608 |
| SP-GQE(n=2,τ=0.5)   | 0.5429  | 0.4751 |
| SP-GQE-i(n=3,τ=0.5) | 0.4712  | 0.4558 |
| GR-RAG              | 0.5287  | 0.4712 |
| GF-RAG              | 0.4995  | 0.4646 |


## Retrieval P@k (title overlap with supporting facts)

- **V-RAG**: 0.760
- **GQE-RAG(n=2)**: 0.690
- **SP-GQE(n=2,τ=0.5)**: 0.700
- **SP-GQE-i(n=3,τ=0.5)**: 0.710
- **GR-RAG**: 0.760
- **GF-RAG**: 0.720

## SP-GQE heatmaps (n × τ)

Answer F1:

f1

Retrieval P@k:

p_at_k

## Positioning vs published RAG / GraphRAG systems


| System                                           | Reference                                              | Notes                                                                   |
| ------------------------------------------------ | ------------------------------------------------------ | ----------------------------------------------------------------------- |
| Vanilla RAG (DPR-style dense retriever + reader) | Lewis et al., NeurIPS 2020; follow-up retrieval stacks | Depends on reader; HotpotQA distractor is harder than full-wiki.        |
| Microsoft GraphRAG (community summaries)         | Edge et al., 2024; msft graphrag                       | Heavy offline indexing; not directly comparable sample-for-sample.      |
| HybGRAG                                          | arXiv:2412.16311 (ACL 2025)                            | Demonstrates hybrid retrieval > single-modality.                        |
| SubgraphRAG                                      | ICLR 2025; arXiv:2410.20724                            | Trained scorer vs our training-free pruning.                            |
| RAG vs GraphRAG systematic study                 | arXiv:2502.11371 (Feb 2025)                            | Motivates hybrid / query-aware graph use (aligned with SP-GQE).         |
| SP-GQE (this work)                               | Repository experiments                                 | Prototype stack: meant for ablations vs baselines under identical code. |


