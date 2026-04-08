# Semantically-Pruned Graph Expansion for Retrieval-Augmented Generation

**A Recall-and-Verify Approach to Knowledge Graph–Guided Query Augmentation**

Master's Dissertation — Internet Systems Engineering

---

## Abstract

Retrieval-Augmented Generation (RAG) grounds large language model responses
in external knowledge, but standard vector-based retrieval cannot capture
relational structure between entities.
Knowledge graph (KG) integration addresses this gap,
yet naive graph expansion introduces noise at an exponential rate
as traversal depth increases.

This dissertation proposes **Semantically-Pruned Graph Query Expansion
(SP-GQE)** — a two-phase method that first expands entities via
knowledge graph traversal (*recall*),
then prunes semantically divergent entities using embedding-based
similarity to query-derived sub-queries and structured graph query
results (*verify*).
An advanced iterative variant (SP-GQE-i) applies pruning at each
traversal hop, producing organically-shaped subgraphs
that grow only along semantically relevant paths.

We evaluate six retrieval strategies on a HotpotQA multi-hop
question answering subset, measuring answer accuracy, faithfulness,
and retrieval precision.
The experimental setup uses entirely free, open-source,
locally-run tools: Ollama, Neo4j Community Edition, FAISS, and spaCy.

---

## 1. Motivation

Standard RAG retrieves document chunks by vector similarity.
This works well for single-fact lookups,
but fails on questions that require connecting information
across multiple documents [1, 2].
A question like *"Who directed the film starring the actor
born in the same city as the inventor of X?"*
demands chaining facts through intermediate entities —
something cosine similarity over flat embeddings cannot do.

Knowledge graphs encode entities and their relationships explicitly,
enabling multi-hop traversal.
Recent surveys [3, 4] and benchmarks [5] confirm growing interest in
combining KGs with RAG (GraphRAG), but also identify a core tension:
graph expansion retrieves structurally connected entities,
many of which are irrelevant to the query.
At *n* hops with average node degree *d*,
the candidate set grows as O(*d*^*n*) — quickly becoming
both computationally expensive and semantically noisy.

Current approaches address this either through trained scorers
(SubgraphRAG [6], requiring model fitting)
or through post-retrieval re-ranking
(Practical GraphRAG [7], which cannot inject missing context).
No existing work proposes a **training-free, pre-retrieval
pruning mechanism** that filters expanded entities
by semantic relevance before they enter the retrieval pipeline.

This gap motivates our research question:

> How does semantically-pruned graph expansion affect
> retrieval quality in RAG, and can iterative per-hop pruning
> make the traversal depth self-regulating?

---

## 2. State of the Art

### 2.1 Retrieval-Augmented Generation

Lewis et al. [1] introduced RAG as a method to reduce hallucination
by retrieving relevant passages from a corpus before generation.
The standard pipeline embeds queries and documents into a shared
vector space, retrieves the top-*k* most similar chunks,
and passes them as context to the language model.

### 2.2 Graph-Enhanced RAG (GraphRAG)

Han et al. [3] define a holistic GraphRAG framework with five
components: query processor, retriever, organizer, generator,
and data source.
Peng et al. [4] formalize the workflow into three stages:
graph-based indexing, graph-guided retrieval,
and graph-enhanced generation.

**Microsoft GraphRAG** [8] extracts entity graphs from corpora
using LLMs, applies hierarchical community detection,
and generates community summaries for global queries.
It is effective for thematic summarization
but expensive in graph construction.

**Self-RAG** [9] (ICLR 2024) trains models to emit reflection tokens
that control *whether* to retrieve.
It addresses retrieval necessity, not retrieval quality —
complementary to our work.

**SubgraphRAG** [6] (ICLR 2025) uses a lightweight MLP
with parallel triple-scoring for subgraph retrieval,
achieving competitive accuracy without fine-tuning the LLM.
It requires training the MLP scorer.

**HybGRAG** [10] (ACL 2025) combines a retriever bank
with a critic module for hybrid textual and relational retrieval,
showing 51% improvement in Hit@1 on the STaRK benchmark.

### 2.3 The Expansion Noise Problem

A systematic evaluation by [5] shows that RAG and GraphRAG have
*distinct* strengths: neither universally dominates.
Graph-expanded retrieval degrades when the expansion introduces
noisy, irrelevant entities.
Entity coverage in constructed KGs reaches only 65.8% on HotpotQA
[11], and GraphRAG underperforms vanilla RAG by 13.4%
on time-sensitive queries where graph information is stale.

### 2.4 Evaluation

RAGAS [12] provides automated, LLM-driven evaluation
of RAG pipelines across faithfulness, relevance,
and context quality, without requiring ground-truth annotations.
HotpotQA [13] is the standard benchmark for multi-hop QA,
providing 113k question-answer pairs with annotated supporting facts.

---

## 3. Proposition

### 3.1 SP-GQE: Recall Then Verify

We propose a two-phase retrieval enhancement:

**Phase 1 — Recall.**
Extract entities from the question (spaCy NER).
Expand each entity to its *n*-hop neighborhood in the KG (Neo4j).
In parallel, decompose the question into semantic sub-queries
and execute structured Cypher queries against the KG,
yielding a compact verified fact set R_Q.

**Phase 2 — Verify.**
Build a probe set from sub-query embeddings and R_Q embeddings.
For each expanded entity, compute its maximum cosine similarity
to any probe.
Entities below threshold τ are pruned.
Survivors and verified facts are injected into an augmented query Q',
which drives the final vector retrieval.

The architecture is shown below:

```
             User Question Q
                    │
        ┌───────────┴────────────┐
        │                        │
        ▼                        ▼
  Entity Extraction       Query Decomposition
  E_Q = {e₁,..eₖ}        S_Q = {q₁,..qₘ}
        │                        │
        ▼                        ▼
  Graph Expansion          Structured Queries
  (Neo4j, n-hop)           (Cypher templates)
  E_exp = E_Q ∪ Nₙ(eᵢ)    R_Q = {r₁,..rₚ}
        │                        │
        └────────┬───────────────┘
                 ▼
        Semantic Pruning
        Probes = emb(S_Q) ∪ emb(R_Q)
        E_pruned = {v : max cos(emb(v), Probes) ≥ τ}
                 │
                 ▼
        Query Augmentation
        Q' = Q ⊕ describe(E_pruned) ⊕ narrate(R_Q)
                 │
                 ▼
        Vector Retrieval (FAISS, top-k)
                 │
                 ▼
        LLM Generation (Ollama)
```

### 3.2 SP-GQE-i: Iterative Controlled Expansion

The batch variant still generates O(*d*^*n*) candidates
before pruning.
We propose an iterative variant where pruning occurs
**at each hop, per node**:

```
frontier ← E_Q
for depth = 1 to n_max:
    candidates ← neighbors(frontier)
    survivors ← {v ∈ candidates : relevance(v) ≥ τ}
    frontier ← survivors
```

Only nodes that survive pruning at hop *k*
are expanded at hop *k+1*.
Failed branches are never explored further.
This produces organically-shaped subgraphs:

```
Batch (expand all, prune after):     Iterative (prune as you grow):

       ●──●──●──✗                          ●──●──●
      /                                   /
 Q──●──●──✗                          Q──●──●──●──●
      \                                   \
       ●──●──●──●──✗                       ●──✗ (stopped)
```

The (n, τ) pair becomes self-regulating:
*n* sets maximum reach, τ governs growth discipline.
Effective depth adapts per branch based on local semantic relevance.

### 3.3 Experimental Design

Six pipelines are compared on 150 HotpotQA questions
(75 multi-hop, 75 single-hop):

| Pipeline | Description |
|----------|-------------|
| V-RAG | Vanilla vector retrieval (baseline) |
| GQE-RAG | Naive graph expansion, no pruning |
| SP-GQE | Batch expansion + semantic pruning |
| SP-GQE-i | Iterative controlled expansion |
| GR-RAG | Post-retrieval graph re-ranking |
| GF-RAG | Graph-filtered document selection |

Key independent variables: hop depth *n* ∈ {1,2,3,5}
and pruning threshold τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.

Evaluation metrics: Answer F1, Exact Match,
Faithfulness, Context Relevance (via RAGAS [12]),
Retrieval Precision@5, pruning ratio, and latency.

Central hypotheses:

1. SP-GQE outperforms naive expansion at n ≥ 2.
2. SP-GQE-i outperforms batch SP-GQE at n ≥ 3.
3. SP-GQE-i at n=5 performs comparably to n=3,
   demonstrating self-regulation.

All tools are open-source and run locally without GPU:
Ollama (Mistral 7B), Neo4j Community Edition (Docker),
FAISS, spaCy, and RAGAS.

---

## References

[1] P. Lewis et al.,
"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,"
*NeurIPS*, 2020.

[2] S. Asai et al.,
"Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection,"
*ICLR*, 2024. arXiv:2310.11511.

[3] J. Han et al.,
"Retrieval-Augmented Generation with Graphs (GraphRAG),"
arXiv:2501.00309, Jan. 2025.

[4] B. Peng et al.,
"Graph Retrieval-Augmented Generation: A Survey,"
arXiv:2408.08921, Aug. 2024.

[5] "RAG vs. GraphRAG: A Systematic Evaluation and Key Insights,"
arXiv:2502.11371, Feb. 2025.

[6] "Simple is Effective: The Roles of Graphs and Large Language
Models in Knowledge-Graph-Based Retrieval-Augmented Generation,"
*ICLR*, 2025. arXiv:2410.20724.

[7] "Towards Practical GraphRAG: Efficient Knowledge Graph
Construction and Hybrid Retrieval at Scale,"
arXiv:2507.03226, Jul. 2025.

[8] Microsoft,
"From Local to Global: A Graph RAG Approach
to Query-Focused Summarization," 2024.
github.com/microsoft/graphrag.

[9] S. Asai et al.,
"Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection,"
*ICLR*, 2024. arXiv:2310.11511.

[10] "HybGRAG: Hybrid Retrieval-Augmented Generation
on Textual and Relational Knowledge Bases,"
*ACL*, 2025. arXiv:2412.16311.

[11] "KG-RAG: Bridging the Gap Between Knowledge and Creativity,"
arXiv:2405.12035, May 2024.

[12] S. Es et al.,
"RAGAs: Automated Evaluation of Retrieval Augmented Generation,"
*EACL*, 2024. arXiv:2309.15217.

[13] Z. Yang et al.,
"HotpotQA: A Dataset for Diverse, Explainable
Multi-Hop Question Answering,"
*EMNLP*, 2018.
