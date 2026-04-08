# Dissertation Plan: Graph-Enhanced Retrieval-Augmented Generation

## Working Title

**"Semantically-Pruned Graph Expansion for Retrieval-Augmented Generation: A Recall-and-Verify Approach to Knowledge Graph–Guided Query Augmentation"**

Alternative titles:

- "Recall Then Verify: Semantic Divergence Pruning for Graph-Expanded Retrieval-Augmented Generation"
- "Lean Graph Expansion for RAG: Mitigating Semantic Drift Through Query-Aware Entity Pruning"

---

# SECTION 1: MOTIVATION

## 1.1 The Problem with Vanilla RAG

Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding
Large Languag
e Model (LLM) responses in external knowledge. However, standard RAG
systems suffer from fundamental limitations:

1. **Flat retrieval** — Vector similarity search treats documents as isolated chunks. It
  cannot reason about *relationships* between entities across chunks. When a question
   requires connecting facts from multiple sources ("Which rocket manufacturer also
   supplied engines for the Artemis program?"), cosine similarity over embeddings
   provides no structural path between the answer entities.
2. **Hallucination under complexity** — When retrieved context is noisy or insufficient,
  LLMs fill gaps with fabricated facts. Studies show GraphRAG underperforms vanilla RAG
   by 13.4% on Natural Questions precisely when the knowledge graph is incomplete
   (arXiv:2502.20854), revealing that *how* you integrate structured knowledge matters
   as much as *whether* you integrate it.
3. **No multi-hop reasoning** — Standard RAG retrieves a fixed top-k set. Multi-hop
  questions require chaining evidence: "Who directed the film starring the actor born
   in the same city as the inventor of X?" Vector search cannot follow such chains.
4. **Opacity** — There is no provenance trail. The user cannot trace *why* a particular
  chunk was retrieved or how the answer was derived.

## 1.2 Why Knowledge Graphs?

Knowledge Graphs (KGs) encode entities as nodes and relationships as edges, enabling
structured traversal. Integrating KGs into RAG addresses the above limitations by:

- Providing **relational pathways** between entities for multi-hop queries
- **Constraining retrieval** to structurally relevant documents, reducing noise
- Offering **explainability** through traceable graph paths
- Enabling **query expansion** — enriching the user's question with related entities
discovered via graph traversal before performing vector search

## 1.3 The Research Gap

Despite growing interest, the field lacks systematic evaluation of *how* graph
information should be integrated into RAG pipelines. The January 2025 comprehensive
survey by Han et al. (arXiv:2501.00309) identifies this as an open challenge. The
February 2025 systematic evaluation by (arXiv:2502.11371) confirms that RAG and
GraphRAG have "distinct strengths across different tasks" — but does not explore
*hybrid* strategies that combine both.

Specifically:

- Most GraphRAG systems either replace vector retrieval with graph retrieval OR use
graphs only for re-ranking *after* vector retrieval.
- Very few works study **graph-guided query expansion** — using graph traversal to
*enrich the query itself* before vector retrieval — as a first-class strategy.
- The effect of **expansion depth** (number of hops) on retrieval quality is
under-explored, despite being critical: too few hops miss relevant context, too many
introduce noise.
- **The expansion noise problem is well-documented but under-addressed.** Literature
reports that unrestricted n-hop expansion produces "bloated subgraphs" with noisy,
irrelevant nodes (see Section 1.1). SubgraphRAG (ICLR 2025) addresses this with a
trained MLP scorer, but this requires training data and model fitting. No existing
work proposes a **training-free, embedding-based pruning mechanism** that filters
expanded entities by their semantic relevance to the query before they enter the
retrieval pipeline.

## 1.4 Motivation Summary (for the dissertation text)

> Vanilla RAG systems rely on flat vector similarity, which cannot capture relational
> structure between entities. Knowledge graph integration promises to address this
> through structured retrieval, but naive graph expansion introduces noise at an
> exponential rate as traversal depth increases. This dissertation proposes
> **Semantically-Pruned Graph Query Expansion (SP-GQE)** — a two-phase
> "recall-and-verify" method that first expands entities via knowledge graph traversal,
> then prunes semantically divergent entities using embedding-based distance to
> query-derived sub-queries. We systematically evaluate pruning threshold and traversal
> depth on multi-hop QA, demonstrating that lean, query-aware expansion outperforms
> both naive expansion and flat vector retrieval.

---

# SECTION 2: STATE OF THE ART

## 2.1 Foundational Concepts

### 2.1.1 Retrieval-Augmented Generation (RAG)

Cite: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
(NeurIPS 2020). The foundational RAG paper. Pipeline: query → embed → retrieve top-k
chunks from vector store → concatenate with query → generate answer with LLM.

### 2.1.2 Knowledge Graphs

Brief background on KGs (RDF triples, property graphs), mention Wikidata, DBpedia,
and domain-specific KGs. Cite standard KG references.

## 2.2 GraphRAG: Surveys and Taxonomies

### Key Survey Papers (all open access):


| #   | Paper                                                                                   | Venue                               | Key Contribution                                                                                                                                          |
| --- | --------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Han et al. "Retrieval-Augmented Generation with Graphs (GraphRAG)"** arXiv:2501.00309 | Survey, Jan 2025                    | Holistic framework: query processor, retriever, organizer, generator, data source. Covers domain-specific GraphRAG. GitHub: github.com/Graph-RAG/GraphRAG |
| 2   | **Peng et al. "Graph Retrieval-Augmented Generation: A Survey"** arXiv:2408.08921       | Survey, Aug 2024                    | First comprehensive GraphRAG survey. Formalizes workflow: graph-based indexing → graph-guided retrieval → graph-enhanced generation                       |
| 3   | **Zhu et al. "Graph-based Approaches and Functionalities in RAG"** arXiv:2504.10499     | Survey, Apr 2025 (revised Jan 2026) | Examines diverse roles of graphs in RAG: database construction, algorithms, pipelines                                                                     |


### How to use in your dissertation:

- Use **Peng et al.** for the canonical 3-stage GraphRAG workflow diagram
- Use **Han et al.** for the 5-component framework and domain taxonomy
- Use **Zhu et al.** for the most recent perspective on graph roles

## 2.3 Key Architectures and Methods

### 2.3.1 Microsoft GraphRAG (2024, open-source)

- **Paper**: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
- **Repo**: github.com/microsoft/graphrag
- **Method**: LLM extracts entities/relations from corpus → builds entity graph →
hierarchical community detection (Leiden algorithm) → community summaries →
map-reduce summarization for global queries
- **Strengths**: Excellent for global/thematic queries over private datasets
- **Limitations**: Expensive graph construction (many LLM calls), optimized for
summarization more than factoid QA
- **Local variant**: github.com/TheAiSingularity/graphrag-local-ollama (runs with
Ollama, no API costs)

### 2.3.2 Self-RAG (ICLR 2024 Oral — top 1%)

- **Paper**: Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection" arXiv:2310.11511
- **Repo**: github.com/AkariAsai/self-rag
- **Method**: Trains LLM to emit special "reflection tokens" that control:
  - Whether to retrieve (adaptive retrieval)
  - Whether retrieved passage is relevant (critique)
  - Whether generated text is faithful to context (self-check)
- **Key insight**: Not all queries need retrieval; over-retrieval hurts quality
- **Relevance to your work**: Self-RAG addresses *when* to retrieve; your work
addresses *how* to retrieve. These are complementary. Discuss as future work /
orthogonal improvement axis.
- **Note**: Self-RAG requires fine-tuned models (7B/13B available). You don't need to
reimplement it — discuss it as a related approach your professor mentioned.

### 2.3.3 SubgraphRAG (ICLR 2025)

- **Paper**: "Simple is Effective: The Roles of Graphs and Large Language Models in
Knowledge-Graph-Based Retrieval-Augmented Generation" arXiv:2410.20724
- **Repo**: github.com/Graph-COM/SubgraphRAG (MIT license)
- **Method**: Lightweight MLP with parallel triple-scoring for subgraph retrieval.
Encodes directional structural distances. Balances retrieval size with LLM capacity.
- **Results**: Competitive with GPT-4o-level accuracy using Llama3.1-8B, no fine-tuning
- **Relevance**: Validates that lightweight graph retrieval can match heavy methods.
Strong "simple is effective" message aligns with your thesis approach.

### 2.3.4 HybGRAG (ACL 2025)

- **Paper**: "HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational
Knowledge Bases" arXiv:2412.16311
- **Method**: Retriever bank + critic module for hybrid textual/relational retrieval.
Agentic refinement with feedback.
- **Results**: 51% relative improvement in Hit@1 on STaRK benchmark
- **Relevance**: Demonstrates that hybrid (vector + graph) retrieval outperforms either
alone — directly supports your thesis hypothesis.

### 2.3.5 G-Retriever (2024)

- **Paper**: arXiv:2402.07630
- **Method**: Formulates graph retrieval as Prize-Collecting Steiner Tree optimization.
Handles graphs exceeding LLM context windows.
- **Relevance**: Addresses subgraph selection — related to your expansion depth parameter.

### 2.3.6 Practical GraphRAG (July 2025)

- **Paper**: "Towards Practical GraphRAG: Efficient Knowledge Graph Construction and
Hybrid Retrieval at Scale" arXiv:2507.03226
- **Method**: Dependency parsing for KG construction (94% of LLM quality, much cheaper).
Hybrid retrieval via Reciprocal Rank Fusion (RRF) of vector + graph scores.
- **Results**: 15% improvement over vector-only baselines
- **Relevance**: RRF fusion is a concrete scoring alternative to your α·sim_vector +
β·sim_graph formula. Consider citing and comparing.

### 2.3.7 KG-RAG with Chain of Explorations (May 2024)

- **Paper**: "KG-RAG: Bridging the Gap Between Knowledge and Creativity" arXiv:2405.12035
- **Method**: Chain of Explorations (CoE) algorithm for traversing KG nodes.
Tested on ComplexWebQuestions dataset.
- **Relevance**: Directly relevant — proposes a traversal strategy. Compare your
hop-based expansion approach against CoE conceptually.

## 2.4 Evaluation Frameworks and Benchmarks

### 2.4.1 RAGAS (EACL 2024)

- **Paper**: "RAGAs: Automated Evaluation of Retrieval Augmented Generation"
arXiv:2309.15217
- **Repo**: github.com/explodinggradients/ragas
- **Metrics**: Faithfulness, Answer Relevance, Context Precision, Context Recall
- **Key advantage**: LLM-as-judge evaluation (can use local Ollama model)
- **You will use this** for automated evaluation in your experiments.

### 2.4.2 Key Comparison Paper

- **"RAG vs. GraphRAG: A Systematic Evaluation and Key Insights"** arXiv:2502.11371
(Feb 2025)
- First systematic head-to-head comparison on text benchmarks
- Finding: Neither universally better; task-dependent trade-offs
- **Crucial for your motivation**: establishes the gap your work fills (hybrid strategies)

### 2.4.3 Datasets


| Dataset                          | Size                                                | Type                  | Why relevant                                                                     |
| -------------------------------- | --------------------------------------------------- | --------------------- | -------------------------------------------------------------------------------- |
| **HotpotQA** (Yang et al., 2018) | Dev: 44MB, 7.4k questions                           | Multi-hop QA          | Gold standard for multi-hop reasoning. Free, CC BY-SA 4.0. Has supporting facts. |
| **Natural Questions** (Google)   | Variable                                            | Single-hop factoid QA | Baseline comparison for single-hop.                                              |
| **ComplexWebQuestions**          | ~34k questions                                      | Complex QA            | Tests compositional reasoning. Used by KG-RAG paper.                             |
| **MS GraphRAG Benchmarks**       | github.com/microsoft/graphrag-benchmarking-datasets | Mixed                 | Official benchmarks from Microsoft.                                              |


**Recommendation for your experiments**: Use a **HotpotQA subset** (100-200 questions
from the distractor dev set). It provides multi-hop questions with gold supporting facts,
is free, lightweight, and is the most-cited benchmark in the papers above.

## 2.5 State-of-the-Art Summary Table


| Method                     | Retrieval Strategy                                | KG Role                                  | Multi-hop?              | Cost                 | Open Source? |
| -------------------------- | ------------------------------------------------- | ---------------------------------------- | ----------------------- | -------------------- | ------------ |
| Vanilla RAG                | Vector similarity (top-k)                         | None                                     | No                      | Low                  | N/A          |
| Microsoft GraphRAG         | Community summaries                               | Entity graph from LLM extraction         | Partial (via summaries) | High (construction)  | Yes          |
| Self-RAG                   | Adaptive (retrieve-or-not)                        | None (uses reflection tokens)            | No                      | Medium (fine-tuning) | Yes          |
| SubgraphRAG                | MLP triple-scoring                                | Existing KG subgraph retrieval           | Yes                     | Low                  | Yes          |
| HybGRAG                    | Hybrid bank + critic                              | Relational KB                            | Yes                     | Medium               | Yes          |
| Practical GraphRAG         | RRF (vector + graph)                              | Dependency-parsed KG                     | Yes                     | Low-Medium           | Yes          |
| **Your Proposal (SP-GQE)** | **Graph-expand → semantic prune → vector search** | **Query expansion + divergence pruning** | **Yes**                 | **Low**              | **Will be**  |


---

# SECTION 3: PROPOSITION — Research Question, Method, and Experiments

## 3.1 Research Question

**Primary RQ:**

> How does semantically-pruned graph expansion affect retrieval quality and answer
> accuracy in RAG systems compared to naive (unpruned) expansion and flat vector
> retrieval?

**Secondary RQs:**

- RQ2: Does semantic pruning mitigate the noise introduced by deeper graph traversal
(n=2, n=3), allowing deeper expansion without performance degradation?
- RQ3: Does graph expansion (pruned or unpruned) help more for multi-hop questions
than single-hop questions?
- RQ4: What is the interaction between pruning threshold (τ) and traversal depth (n)
— is there an optimal (n, τ) pair?
- RQ5: How does the pruned expansion approach compare to post-retrieval graph
re-ranking as an integration strategy?

## 3.2 Proposed Method: Semantically-Pruned Graph Query Expansion (SP-GQE)

### 3.2.0 Core Intuition — "Recall Then Verify"

The method is inspired by how human associative memory works:

1. **Recall phase (Graph Expansion):** When you try to remember something, your brain
  follows associative links freely — "I was walking to the lake after I came from
   church that day." This is broad, fast, and uncritical. In our pipeline, this
   corresponds to n-hop graph traversal: cast a wide net over structurally connected
   entities.
2. **Verify phase (Semantic Pruning + Structured Queries):** Your critical faculty
  then checks each recalled association against known constraints — "No, it was
   Monday, so I couldn't have gone to church." You verify against both your general
   sense of the situation (sub-query embeddings) and specific facts you can look up
   (structured graph query results). Associations that don't hold up are discarded.
   In our pipeline, this corresponds to computing semantic divergence between each
   expanded entity and a combined probe set of natural language sub-queries (S_Q)
   and their structured Cypher answers (R_Q), pruning entities that are structurally
   connected but semantically irrelevant. The verified facts (R_Q) are then also
   carried forward to enrich the final query — they're compact but high-value.

This two-phase approach allows deeper graph traversal (more recall) without
proportional noise increase (verification keeps it lean), while the structured
query results do double duty: sharpening the pruning AND enriching the retrieval.

### 3.2.1 Architecture Overview

```
             ┌─────────────────────────────────┐
             │         User Question Q          │
             └──────────┬──────────────────────┘
                        │
            ┌───────────┴───────────────┐
            │                           │
            ▼                           ▼
   ┌────────────────────┐   ┌───────────────────────┐
   │  Entity Extraction │   │  Query Decomposition   │
   │  (spaCy NER)       │   │  (NLP / LLM)           │
   │                    │   │                         │
   │  E_Q = {e₁,..eₖ}  │   │  S_Q = {q₁, q₂,..qₘ}  │
   └────────┬───────────┘   └─────┬─────────────────┘
            │                     │
            ▼                     ▼
   ┌────────────────────┐   ┌───────────────────────┐
   │  Graph Expansion   │   │  Structured Graph      │
   │  (Neo4j n-hop)     │   │  Queries (Cypher)      │
   │                    │   │                         │
   │  E_exp = E_Q ∪     │   │  For each qⱼ ∈ S_Q:    │
   │    ⋃ Nₙ(eᵢ)       │   │   → Cypher query on     │
   │                    │   │     entities in E_Q      │
   │  (broad, uncritical│   │   → R_Q = {r₁,..rₚ}    │
   │   associative net) │   │     (compact fact set)   │
   └────────┬───────────┘   └─────┬─────────────────┘
            │                     │
            ▼                     ▼
   ┌──────────────────────────────────────────────┐
   │          SEMANTIC PRUNING                     │
   │                                               │
   │  Probes = emb(S_Q) ∪ emb(R_Q)                │
   │  (sub-queries + their structured answers)     │
   │                                               │
   │  For each candidate v ∈ E_exp:                │
   │   relevance(v) = max cos(emb(v), Probes)      │
   │                                               │
   │  E_pruned = {v : relevance(v) ≥ τ}            │
   └──────────────────┬───────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────┐
   │          QUERY AUGMENTATION                   │
   │                                               │
   │  Q' = Q                                       │
   │     ⊕ " Graph context: " ⊕ describe(E_pruned)│
   │     ⊕ " Verified facts: " ⊕ narrate(R_Q)     │
   │                                               │
   │  (Both pruned entities AND structured query   │
   │   results feed into the augmented query)       │
   └──────────────────┬───────────────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  Vector Retrieval     │
            │  FAISS top-k using Q' │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  LLM Generation       │
            │  (Ollama / Mistral)   │
            └──────────────────────┘
```

### 3.2.2 Formal Definition

**Inputs:**

- Question Q with extracted entity set E_Q = {e₁, e₂, ..., eₖ}
- Knowledge Graph G = (V, E) stored in Neo4j
- Hop depth parameter n ∈ {1, 2, 3}
- Pruning threshold τ ∈ [0, 1]
- Embedding function emb() (nomic-embed-text via Ollama)

---

**Phase 1a — Recall (Graph Expansion):**
For each entity e ∈ E_Q, retrieve its n-hop neighborhood:
  Nₙ(e) = {v ∈ V : shortest_path(e, v) ≤ n}

Expanded entity set (union of all neighborhoods):
  E_expanded = E_Q ∪ ⋃(e ∈ E_Q) Nₙ(e)

**Phase 1b — Query Decomposition:**
Generate a set of semantic sub-queries from Q:
  S_Q = {q₁, q₂, ..., qₘ} = decompose(Q)

Sub-queries capture different semantic facets of the question. Three strategies
(in order of simplicity):

  a) **Noun-phrase extraction** (spaCy): extract noun chunks from Q.
     Example: "Which missions used rockets built by private companies?"
     → {"missions", "rockets", "private companies"}

  b) **Entity-centric sub-queries**: for each entity in E_Q, form
     "What is [entity]?" and "[entity] in context of Q"

  c) **LLM decomposition**: prompt the LLM to split Q into atomic sub-questions.
     Example: → {"What missions exist?", "What rockets were used?",
     "Which companies are private?", "Which companies built rockets?"}

  Recommendation: Start with (a) for speed. Use (c) as an ablation comparison.

**Phase 1c — Structured Graph Queries:**
For each sub-query qⱼ ∈ S_Q, compose and execute a Cypher query against the KG
using entities from E_Q as anchor points. Collect the results into a compact
fact set R_Q.

  R_Q = {r₁, r₂, ..., rₚ} = ⋃(j=1..m) cypher_query(qⱼ, E_Q, G)

  Each rᵢ is a short factual statement derived from a graph pattern match.

  **Cypher query construction** (template-based, not free-form):

  For each sub-query qⱼ that mentions entity eᵢ ∈ E_Q, generate a Cypher
  query that retrieves direct relationships of eᵢ relevant to qⱼ's focus.

  Example:
    Q: "Which missions used rockets built by private companies?"
    E_Q: {Artemis I}
    Sub-query q₁: "missions"  →  already the anchor, skip
    Sub-query q₂: "rockets"   →
      MATCH (m:Entity {name: "Artemis I"})-[r]->(target)
      WHERE type(r) CONTAINS "USE" OR type(r) CONTAINS "ROCKET"
      RETURN m.name, type(r), target.name LIMIT 5

```
Result r₁: "Artemis I -[USES]-> Space Launch System"
```

  The query templates are simple and few (3-4 patterns):
    1. Direct relations:  (e)-[r]->(?) or (?)-[r]->(e)
    2. Typed neighbors:   (e)-[r]->(?) WHERE type(r) matches sub-query keyword
    3. Two-hop paths:     (e)-[]->(mid)-[]->(?) for bridge questions

  **Handling empty results**: If a Cypher query returns nothing, that sub-query
  simply contributes nothing to R_Q. No harm — R_Q is additive.

  **Why this is lightweight**: These are 3-5 parameterized Cypher templates with
  entity names slotted in. No SPARQL endpoint, no query planner, no reasoning
  engine — just Neo4j pattern matching, which returns in milliseconds.

---

**Phase 2 — Verify (Semantic Pruning):**
For each candidate entity v ∈ E_expanded \ E_Q:

  Build the pruning probe set by combining sub-query embeddings with structured
  result embeddings:

```
Probes = {emb(qⱼ) : qⱼ ∈ S_Q} ∪ {emb(rᵢ) : rᵢ ∈ R_Q}
```

  This is the key enrichment: the Cypher results R_Q provide high-precision
  semantic anchors alongside the broader sub-queries S_Q. A sub-query like
  "rockets" is vague; a Cypher result like "Artemis I uses Space Launch System"
  is specific. Together they form a tighter relevance envelope.

  Compute entity relevance as the maximum cosine similarity between v's
  embedding and any probe embedding:

```
relevance(v) = max(p ∈ Probes) cos_sim(emb(label(v)), p)
```

  Where label(v) returns the textual label/description of node v from the KG.

  Pruning rule:
    E_pruned = E_Q ∪ {v ∈ E_expanded \ E_Q : relevance(v) ≥ τ}

  Note: Original question entities E_Q are always kept (never pruned).

  The **divergence** of an entity is defined as:
    divergence(v) = 1 - relevance(v)

  Entities with high divergence are structurally connected in the KG but
  semantically unrelated to the question — these are the "false memories"
  that get pruned.

  **Effect of R_Q on pruning quality**: Without R_Q, pruning compares entity
  embeddings against short noun phrases ("rockets", "companies") — coarse
  semantic matching. With R_Q, pruning also compares against precise graph
  facts ("Space Launch System manufactured by Boeing") — entities semantically
  close to *verified graph relationships* are more likely to survive. This
  makes pruning both more selective (removes noise better) and more permissive
  of genuinely relevant entities (reduces false pruning).

---

**Phase 3 — Augmented Retrieval:**
Construct augmented query by combining three information sources:

  Q' = Q
     ⊕ " Graph context: " ⊕ describe(E_pruned)
     ⊕ " Verified facts: " ⊕ narrate(R_Q)

Where:

- describe() converts graph triples involving pruned entities to natural language
(e.g., "(Artemis I, USES, SLS)" → "Artemis I uses Space Launch System")
- narrate() converts structured query results R_Q to natural language
(e.g., r₁ = "Artemis I -[USES]-> Space Launch System"
 → "Artemis I uses the Space Launch System rocket")

**Why inject R_Q into Q' (not just use it for pruning)?**

The structured query results are:

- **Compact**: typically 3-10 short factual statements (< 200 tokens total)
- **High-precision**: derived from exact graph pattern matches, not fuzzy expansion
- **Potentially disproportionately informative**: they capture the direct relational
answers to the sub-queries — exactly the kind of bridging facts that multi-hop
questions require and that vector retrieval alone struggles to surface

The cost of including them in Q' is negligible (they're shorter than a single
retrieved document chunk), but they may anchor the vector retrieval toward
passages that discuss these specific relationships, and later help the LLM
generate a grounded answer even if the top-k retrieved passages are imperfect.

Use Q' as the query for FAISS vector search over document embeddings.
Retrieve top-k documents and pass to the LLM for answer generation.

### 3.2.3 Why This Works (Computational Argument)

The exponential expansion problem:

- A graph with average degree d has O(dⁿ) entities at n hops.
- At d=10, n=3: ~1,000 candidate entities → 1,000 embeddings to compute and
inject into the query. This is slow AND noisy.

Semantic pruning reduces this:

- Embedding computation for pruning candidates is cheap: a single batch call to
nomic-embed-text (384-dim vectors, ~0.1ms per embedding on CPU).
- The probe set is small: |S_Q| + |R_Q| ≈ 5 sub-queries + 5 Cypher results = ~10
probe embeddings. Cosine similarity is O(10 × |E_expanded|). For 1,000
candidates = 10,000 dot products. Negligible.
- With a reasonable τ, pruning typically retains 10-30% of candidates, reducing
the augmented query to a manageable size.

The structured query overhead is minimal:

- Cypher queries against a small Neo4j graph (2-5k nodes) return in < 10ms each.
- 3-5 template queries per question = ~30-50ms total. Comparable to a single
LLM embedding call.
- R_Q adds ~100-200 tokens to Q'. For context: the LLM generation step processes
2,000-4,000 tokens of retrieved documents. The R_Q overhead is < 5% of that.

Net effect: You can safely expand to n=2 or n=3 hops and let pruning handle
the noise, while the Cypher results provide high-precision semantic anchors that
make the pruning sharper AND enrich the final query with verified graph facts.

### 3.2.5 Advanced Variant: Iterative Controlled Expansion (SP-GQE-i)

The batch SP-GQE architecture described above has a structural inefficiency:
it expands *all* n-hop neighbors first, then prunes the full set at once.
This means at n=3 with average degree d=10, you still compute and embed ~1,000
candidate entities before discarding most of them.

**Insight:** The (n, τ) pair can be made **self-regulating** if pruning is
applied iteratively at each hop, per node, during the expansion itself. Instead
of "expand everything, then prune," the traversal becomes "expand one hop,
prune, then only expand surviving nodes further."

**Algorithm: Controlled Expansion (SP-GQE-i)**

```
function controlled_expand(E_Q, Probes, G, n_max, τ):
    frontier = E_Q           # start with question entities
    E_result = E_Q            # always keep originals
    for depth = 1 to n_max:
        candidates = neighbors(frontier, G)  # 1-hop from current frontier
        # Prune: only keep semantically relevant candidates
        survivors = {v ∈ candidates : relevance(v, Probes) ≥ τ}
        E_result = E_result ∪ survivors
        frontier = survivors  # ONLY survivors seed the next hop
    return E_result
```

**Key difference from batch SP-GQE:**

- In batch SP-GQE: all dⁿ candidates are generated, then filtered by τ.
- In SP-GQE-i: pruning happens *at each hop*. A node that fails τ at hop 1
is never expanded further — its entire subtree is eliminated.

**Biological analogy — root growth:**
This behaves like fungi mycelium or tree roots searching for nutrients/water.
Growth extends along paths where the environment signals relevance (soil
nutrients = semantic similarity to query). Branches that reach barren ground
(high divergence) stop growing. Branches that hit rich ground (low divergence)
keep extending. The result is an organically shaped, focused subgraph rather
than a uniform sphere of n-hop radius.

```
Batch SP-GQE (expand all, then prune):      SP-GQE-i (prune as you grow):

        ●───●───●───✗                              ●───●───●
       /                                           /
  Q───●───●───✗                               Q───●───●───●───●
       \                                           \
        ●───●───●───●───✗                           ●───✗ (stopped)
       
  Expands full sphere, prunes after.       Grows only along relevant paths.
  Wasteful at high n.                      Naturally lean at any n.
```

**Implications:**

- n_max can be set higher (e.g., 4 or 5) without fear of explosion, because
τ controls growth at each step. The effective depth is *adaptive* — some
branches go deep, others stop at hop 1.
- The (n, τ) pair becomes truly self-regulating: n sets the maximum reach,
τ sets the growth discipline. Together they produce a query-dependent subgraph
whose shape reflects the semantic structure of the question.
- Computational cost is bounded by the number of surviving entities per hop,
not by dⁿ. In practice, if τ retains ~20% at each hop: d=10, τ→2 survivors
per node, n=3 → ~8 entities traversed total, not 1,000.

**Implementation cost:** Minimal — it's the same pruning function called inside
a loop instead of after expansion. The Cypher query changes from "all n-hop
neighbors" to "1-hop neighbors of this set" called n times.

**Experimental role:** SP-GQE-i is tested as an additional pipeline to determine
whether iterative pruning outperforms batch pruning at higher depths (n ≥ 3).

### 3.2.6 Comparison Methods (Experimental Pipelines)

You will implement and compare **6 pipelines**:


| #   | Pipeline                           | Abbreviation | Description                                                                                            |
| --- | ---------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------ |
| 1   | **Baseline**                       | V-RAG        | Vanilla RAG: question → FAISS top-k → LLM                                                              |
| 2   | **Naive Graph Expansion**          | GQE-RAG      | Expand query via KG (no pruning) → vector search → LLM                                                 |
| 3   | **Batch Pruned Expansion**         | SP-GQE       | Expand all n-hop → batch prune by divergence → vector search → LLM                                     |
| 4   | **Iterative Controlled Expansion** | SP-GQE-i     | **Advanced variant**: prune per-node at each hop, grow only along relevant paths → vector search → LLM |
| 5   | **Graph Re-ranking**               | GR-RAG       | Vanilla retrieval → re-rank by graph proximity → LLM                                                   |
| 6   | **Graph Filter**                   | GF-RAG       | Retrieve only docs mentioning KG-connected entities → LLM                                              |


This gives you a clean **ablation study** that isolates four questions:

- Does graph expansion help at all? (V-RAG vs. GQE-RAG)
- Does pruning improve expansion? (GQE-RAG vs. SP-GQE) ← **core contribution**
- Does iterative pruning outperform batch pruning? (SP-GQE vs. SP-GQE-i) ← **advanced contribution**
- Where should the graph act — pre-retrieval or post-retrieval? (SP-GQE vs. GR-RAG)

## 3.3 Experimental Design

### 3.3.1 Constraints-Aware Design


| Constraint              | Solution                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| No paid tools           | Ollama (free local LLM), Neo4j Community (free, Docker), FAISS (free), spaCy (free)      |
| No GPU                  | Ollama runs CPU-only with quantized models (Mistral 7B Q4, ~4GB RAM)                     |
| Work laptop safety      | Everything in Docker containers (Neo4j) + Python venv. No system-level installs.         |
| Limited time (<4h/week) | Pre-built dataset (HotpotQA), small sample (150 questions), automated evaluation (RAGAS) |
| Limited energy          | Modular pipeline — build one component per week, automate evaluation                     |


### 3.3.2 Technology Stack

```
┌─────────────────────────────────────────────┐
│                 Docker Compose               │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │   Neo4j CE   │  │  Python 3.10+ venv   │  │
│  │  (graph DB)  │  │  ┌────────────────┐  │  │
│  │  Port: 7474  │  │  │  LangChain     │  │  │
│  │  Port: 7687  │  │  │  FAISS         │  │  │
│  └─────────────┘  │  │  spaCy          │  │  │
│                    │  │  RAGAS          │  │  │
│  ┌─────────────┐  │  │  neo4j driver   │  │  │
│  │   Ollama     │  │  └────────────────┘  │  │
│  │  (local LLM) │  └──────────────────────┘  │
│  │  Mistral 7B  │                            │
│  └─────────────┘                             │
└─────────────────────────────────────────────┘
```

**Specific versions / models:**

- **LLM**: Mistral 7B (via Ollama) — good balance of quality and speed on CPU
- **Embeddings**: `nomic-embed-text` via Ollama (384-dim, fast on CPU)
- **Graph DB**: Neo4j Community Edition 5.x (Docker: `neo4j:5-community`)
- **Vector Store**: FAISS (faiss-cpu package)
- **NER**: spaCy `en_core_web_sm` (or `en_core_web_trf` if laptop handles it)
- **Evaluation**: RAGAS with Ollama as the judge LLM
- **Orchestration**: Plain Python scripts (LangChain optional, adds complexity)

### 3.3.3 Dataset Preparation

**Source**: HotpotQA distractor dev set (44MB, 7,405 questions)

**Sampling strategy** (minimal data processing):

1. Download HotpotQA dev set (single JSON file from hotpotqa.github.io)
2. Filter for "hard" and "medium" difficulty bridge-type questions (multi-hop)
3. Sample 150 questions:
  - 75 multi-hop (bridge) questions
  - 75 single-hop comparison questions
4. For each question, HotpotQA provides:
  - Gold answer
  - Supporting facts (paragraph title + sentence index)
  - 10 context paragraphs (2 relevant + 8 distractors)

**Knowledge Graph Construction** (the lightweight way):

- For the 150 sampled questions, take all associated Wikipedia paragraphs (~1500 paragraphs)
- Run spaCy NER to extract entities from each paragraph
- Use Ollama (Mistral) to extract (subject, relation, object) triples from each paragraph
with a simple prompt:
  ```
  Extract entity-relationship triples from this text.
  Format: (Subject, Relation, Object)
  Text: {paragraph}
  ```
- Load triples into Neo4j
- Expected size: ~2,000-5,000 nodes, ~3,000-8,000 edges
- **Estimated time**: ~2-3 hours (one-time, automated script)

### 3.3.4 Experimental Protocol

**Independent Variables**: Pipeline type × Hop depth (n) × Pruning threshold (τ)

**Experimental Matrix:**


| Exp     | Pipeline     | Hop (n) | Threshold (τ) | Description                            |
| ------- | ------------ | ------- | ------------- | -------------------------------------- |
| E1      | V-RAG        | —       | —             | Baseline: pure vector retrieval        |
| E2a     | GQE-RAG      | 1       | — (no prune)  | Naive expansion, 1-hop                 |
| E2b     | GQE-RAG      | 2       | — (no prune)  | Naive expansion, 2-hop                 |
| E2c     | GQE-RAG      | 3       | — (no prune)  | Naive expansion, 3-hop                 |
| **E3a** | **SP-GQE**   | **1**   | **0.3**       | Batch pruned, 1-hop, loose             |
| **E3b** | **SP-GQE**   | **2**   | **0.3**       | Batch pruned, 2-hop, loose             |
| **E3c** | **SP-GQE**   | **2**   | **0.5**       | Batch pruned, 2-hop, medium            |
| **E3d** | **SP-GQE**   | **2**   | **0.7**       | Batch pruned, 2-hop, strict            |
| **E3e** | **SP-GQE**   | **3**   | **0.5**       | Batch pruned, 3-hop, medium            |
| **E4a** | **SP-GQE-i** | **3**   | **0.5**       | Iterative controlled, max 3-hop        |
| **E4b** | **SP-GQE-i** | **5**   | **0.5**       | Iterative controlled, max 5-hop        |
| **E4c** | **SP-GQE-i** | **3**   | **0.3**       | Iterative controlled, max 3-hop, loose |
| E5      | GR-RAG       | 1       | —             | Post-retrieval graph re-ranking        |
| E6      | GF-RAG       | 1       | —             | Graph-filtered document selection      |


Notes:

- SP-GQE experiments focus on n=2 with varying τ because batch pruning's
contribution is most visible there.
- SP-GQE-i experiments test higher n_max values (3 and 5) because iterative
pruning makes deeper traversal practical. E4b (n_max=5) would be infeasible
with batch expansion but is cheap with iterative pruning.
- Comparing E3e (batch, n=3, τ=0.5) vs. E4a (iterative, n=3, τ=0.5) directly
isolates the effect of iterative vs. batch pruning at the same parameters.

**Fixed parameters:**

- Top-k = 5 retrieved documents
- LLM: Mistral 7B (temperature=0 for reproducibility)
- Embedding model: nomic-embed-text (384-dim)
- Query decomposition: noun-phrase extraction via spaCy (baseline strategy)
- Same 150 questions across all experiments

**Evaluation Metrics:**


| Metric                    | How measured                                                | Tool                                   |
| ------------------------- | ----------------------------------------------------------- | -------------------------------------- |
| **Answer F1**             | Token-level F1 against HotpotQA gold answer                 | Custom script (standard HotpotQA eval) |
| **Exact Match (EM)**      | Binary: predicted answer matches gold                       | Custom script                          |
| **Faithfulness**          | Is answer supported by retrieved context?                   | RAGAS (LLM-as-judge via Ollama)        |
| **Context Relevance**     | Are retrieved passages relevant to question?                | RAGAS                                  |
| **Retrieval Precision@5** | How many of top-5 docs are actually relevant?               | Compare vs. HotpotQA supporting facts  |
| **Expansion Size**        | |E_pruned| / |E_expanded| — pruning ratio                   | Logged per question                    |
| **Latency**               | Wall-clock time per query (expansion + pruning + retrieval) | Python timer                           |


The last two metrics are unique to your contribution: they quantify the
computational efficiency gains from pruning.

### 3.3.5 Expected Outcomes and Hypotheses


| ID      | Hypothesis                                            | Expected Result                          | Why                                                                                                           |
| ------- | ----------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| H1      | GQE-RAG (n=1) > V-RAG on multi-hop                    | Higher F1/Faithfulness                   | Graph expansion adds relational context                                                                       |
| H2      | GQE-RAG (n=3) < GQE-RAG (n=1)                         | Degradation at depth 3                   | Noise from exponential expansion                                                                              |
| **H3**  | **SP-GQE (n=2, τ=0.5) > GQE-RAG (n=2)**               | **Higher F1, lower noise**               | **Pruning removes irrelevant entities**                                                                       |
| **H4**  | **SP-GQE (n=3, τ=0.5) ≈ SP-GQE (n=2, τ=0.5)**         | **Batch pruning rescues deep expansion** | **Deeper traversal recovers useful entities that survive pruning**                                            |
| H5      | SP-GQE benefit is larger on multi-hop than single-hop | Bigger delta on bridge questions         | Multi-hop questions have more entities to expand and prune                                                    |
| H6      | Pruning ratio increases with n                        | More entities pruned at deeper hops      | Distant entities are less likely to be semantically relevant                                                  |
| **H7**  | **SP-GQE (n=2, τ=0.5) is faster than GQE-RAG (n=2)**  | **Lower latency**                        | **Smaller augmented query → faster embedding + retrieval**                                                    |
| **H8**  | **SP-GQE-i (n=3) ≥ SP-GQE (n=3) at same τ**           | **Iterative ≥ batch at same depth**      | **Per-node pruning avoids expanding irrelevant subtrees, keeping signal cleaner**                             |
| **H9**  | **SP-GQE-i (n=5) ≈ SP-GQE-i (n=3)**                   | **Deeper reach without degradation**     | **Iterative pruning makes n_max a soft ceiling; growth self-terminates when branches hit semantic dead ends** |
| **H10** | **SP-GQE-i is faster than SP-GQE at n ≥ 3**           | **Lower latency**                        | **Fewer total candidates generated (pruned subtrees never expanded)**                                         |


**Key story (three acts):**

1. **H3**: Batch pruning improves over naive expansion.
2. **H8**: Iterative pruning improves over batch pruning at deeper traversals.
3. **H9**: Iterative pruning makes the depth parameter nearly irrelevant — growth
  naturally self-terminates, so n_max=5 produces similar results to n_max=3 because
   irrelevant branches die early and relevant branches exhaust their neighborhoods.

If H9 holds, it's the strongest finding: the (n, τ) pair is self-regulating.
If H9 fails (n=5 degrades), that still establishes the limits of controlled
expansion and is a valid result.

### 3.3.6 Ablation Studies

1. **Pruning threshold sensitivity**: Fix n=2, sweep τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.
  Plot F1/EM vs. τ. Expect inverted-U: too loose (τ=0.1) keeps noise, too strict
   (τ=0.9) over-prunes useful entities.
2. **Hop depth × pruning interaction**: 2D heatmap of F1 over (n, τ) grid.
  This is the most visually compelling result for the dissertation.
3. **Batch vs. iterative pruning at depth**: Compare SP-GQE vs. SP-GQE-i at
  n=3 and n=5 with τ=0.5. Report both accuracy and entities-traversed count.
   This directly tests whether iterative pruning is self-regulating.
4. **Query decomposition strategy**: On a 30-question subset, compare:
  - (a) Noun-phrase extraction (spaCy) — baseline
  - (c) LLM decomposition (Mistral prompt) — richer but slower
   Report impact on pruning quality. This justifies or rejects the need for LLM
   in the decomposition step.
5. **Pruning ratio analysis**: For each (n, τ), report mean/median/std of
  |E_pruned|/|E_expanded|. Show how aggressively pruning acts at each depth.
   For SP-GQE-i, additionally report *effective depth* — the average maximum hop
   reached across surviving branches (this may be << n_max).
6. **Entity extraction method**: Compare spaCy NER vs. LLM-based extraction on
  20 questions. Discuss impact without full re-run.

### 3.3.7 Visualization: Animated Expansion Traces

For a select set of example questions (3-5), produce an animated visual aid
comparing expansion behavior across methods:

**Frame sequence (per question):**

- Frame 0: Question entities (E_Q) highlighted on the KG subgraph
- Frame 1: Hop 1 expansion (all candidates shown)
- Frame 2: After pruning at hop 1 (pruned nodes grayed out / removed)
- Frame 3: Hop 2 expansion from survivors only (SP-GQE-i) or full (SP-GQE)
- Frame 4: After pruning at hop 2
- ...continue to n_max
- Final frame: Overlay ground truth entities (from HotpotQA supporting facts)
in a different color to show coverage

**What this shows:**

- SP-GQE-i's expansion visually resembles root/mycelium growth — branching
along semantically fertile paths and terminating in barren directions.
- Naive GQE-RAG produces a uniform sphere.
- Batch SP-GQE produces a sphere-then-filter.
- The ground truth overlay demonstrates whether the organic growth pattern
actually captures the right entities.

**Implementation:** Neo4j Browser or a simple Python graph visualization
(networkx + matplotlib animation). Export as GIF or MP4 for the dissertation
and defense presentation. This takes ~2-3 hours for 3-5 examples.

**Academic value:** This is not just illustration — it's qualitative evidence
of the method's behavior. Professors respond strongly to clear visual
demonstrations of algorithmic behavior, and the root-growth analogy becomes
immediately tangible when animated.

## 3.4 Implementation Timeline

**Total budget: ~64 hours (4h/week × 16 weeks, mid-Feb to mid-June)**


| Week  | Hours | Task                                                                          | Deliverable                                              |
| ----- | ----- | ----------------------------------------------------------------------------- | -------------------------------------------------------- |
| 1-2   | 8h    | Environment setup: Docker Compose (Neo4j + Ollama), Python venv, install deps | Working `docker-compose.yml`, Python env, Ollama running |
| 3-4   | 8h    | Data preparation: Download HotpotQA, sample 150 questions, extract entities   | `data/` folder with sampled questions + spaCy entities   |
| 5-6   | 8h    | KG construction: Run triple extraction via Ollama, load into Neo4j, verify    | Populated Neo4j graph, basic Cypher queries working      |
| 7-8   | 8h    | Implement V-RAG baseline: FAISS index, embedding pipeline, Ollama generation  | Working baseline, first Answer F1 numbers                |
| 9     | 4h    | Implement GQE-RAG (naive): graph expansion + query augmentation (no pruning)  | GQE-RAG working for n=1,2,3                              |
| 10    | 4h    | **Implement SP-GQE (batch)**: query decomposition + Cypher queries + pruning  | SP-GQE working with tunable τ                            |
| 11    | 4h    | **Implement SP-GQE-i (iterative)**: refactor pruning into per-hop loop        | SP-GQE-i working; GF-RAG and GR-RAG variants             |
| 12    | 4h    | Run full evaluation: all 14 experiments, RAGAS metrics, log pruning ratios    | Results CSV, raw metrics                                 |
| 13    | 4h    | Run ablation studies: τ sweep, batch vs. iterative, decomposition comparison  | Ablation tables and heatmap data                         |
| 14    | 4h    | Generate animated expansion traces (3-5 examples) + static figures            | GIFs/MP4s, heatmap, F1 curves                            |
| 15-16 | 8h    | Write results + discussion chapter                                            | Draft results section                                    |
| 17    | 4h    | Final polish, limitations, conclusion, abstract                               | Complete draft                                           |


**Total: ~68 hours (17 weeks at 4h/week, mid-Feb to mid-June)**

**Buffer / priority notes:**

- SP-GQE-i (week 11) is a small refactor of SP-GQE — same pruning function called
inside a hop loop instead of after full expansion. ~30 min of code changes.
- The animated visualizations (week 14) are high-impact for the defense presentation
but can be cut if time is tight without affecting the written results.
- If weeks slip, drop ablation items 4 and 6 from Section 3.3.6 first (query
decomposition comparison and entity extraction comparison). The core story
(batch vs. iterative, τ sensitivity, heatmap) must survive.

**Implementation complexity of the pruning + enrichment layer:**

```python
# Pseudocode for the complete verify-and-enrich step (~80 lines in practice)

def structured_graph_queries(sub_queries, entity_names, neo4j_driver):
    """Phase 1c: Run Cypher templates for each sub-query × entity pair."""
    results = []
    for q in sub_queries:
        for entity in entity_names:
            # Template: direct relationships matching sub-query keywords
            cypher = """
                MATCH (e:Entity {name: $name})-[r]->(t)
                RETURN e.name, type(r), t.name LIMIT 5
            """
            rows = neo4j_driver.run(cypher, name=entity)
            for row in rows:
                results.append(f"{row['e.name']} {row['type(r)']} {row['t.name']}")
    return results  # R_Q: list of short fact strings

def semantic_prune(expanded_entities, sub_queries, graph_results, threshold, embed_fn):
    """Phase 2: Prune using both S_Q and R_Q as probes."""
    # Embed all candidate entity labels (batch call)
    entity_embeddings = embed_fn([e.label for e in expanded_entities])
    # Build probe set: sub-queries + structured graph results
    probes = sub_queries + graph_results  # S_Q ∪ R_Q
    probe_embeddings = embed_fn(probes)
    # Compute pairwise cosine similarity matrix
    # Shape: (num_entities, num_probes)
    sim_matrix = cosine_similarity(entity_embeddings, probe_embeddings)
    # Max similarity across ALL probes for each entity
    max_relevance = sim_matrix.max(axis=1)
    # Keep entities above threshold
    pruned = [e for e, r in zip(expanded_entities, max_relevance) if r >= threshold]
    return pruned

def augment_query(question, pruned_entities, graph_results):
    """Phase 3: Build Q' from three sources."""
    graph_context = "; ".join(describe(e) for e in pruned_entities)
    verified_facts = "; ".join(graph_results)
    return f"{question} Graph context: {graph_context} Verified facts: {verified_facts}"
```

**SP-GQE-i iterative variant** (refactors the expansion, reuses the same pruning):

```python
def controlled_expand(entity_names, probes, neo4j_driver, n_max, threshold, embed_fn):
    """SP-GQE-i: prune at each hop, only expand survivors."""
    probe_embeddings = embed_fn(probes)
    frontier = set(entity_names)  # current growth tips
    result = set(entity_names)    # accumulated survivors

    for depth in range(1, n_max + 1):
        # Get 1-hop neighbors of current frontier
        candidates = get_neighbors(frontier, neo4j_driver)
        candidates -= result  # skip already-included entities
        if not candidates:
            break  # no more growth possible

        # Prune: same function as batch SP-GQE
        entity_embs = embed_fn([c.label for c in candidates])
        sim_matrix = cosine_similarity(entity_embs, probe_embeddings)
        max_rel = sim_matrix.max(axis=1)
        survivors = {c for c, r in zip(candidates, max_rel) if r >= threshold}

        result |= survivors
        frontier = survivors  # ONLY survivors seed the next hop

    return result
```

This is the core of your contribution — and it remains trivially implementable.
The iterative variant is a ~20-line refactor of the batch version.
The Cypher templates are 3-4 parameterized patterns, not a query generation engine.

## 3.5 Contribution Statement

This dissertation makes the following contributions:

1. **Semantically-Pruned Graph Query Expansion (SP-GQE):** A novel, training-free
  method that combines knowledge graph traversal with embedding-based semantic
   filtering to produce lean, query-relevant entity expansions. Unlike SubgraphRAG
   (ICLR 2025) which requires a trained MLP, SP-GQE uses only pre-computed embeddings
   and cosine similarity — making it reproducible with zero training cost.
2. **Dual-source pruning and enrichment:** The pruning probes combine both natural
  language sub-queries (S_Q) and structured Cypher query results (R_Q), yielding
   tighter relevance filtering than either source alone. The same R_Q results are
   injected into the augmented query as "verified facts" — compact, high-precision
   graph-derived statements that anchor retrieval and generation at negligible token cost.
3. **Iterative Controlled Expansion (SP-GQE-i):** An advanced variant that applies
  semantic pruning per-node at each hop during traversal, producing organically-shaped
   subgraphs that grow only along semantically relevant paths. This makes the (n, τ)
   pair self-regulating: n sets maximum reach, τ controls growth discipline, and the
   expansion naturally self-terminates when branches hit semantic dead ends.
4. **The "Recall-and-Verify" framework:** A conceptual contribution framing graph
  expansion as associative recall and semantic pruning as critical verification,
   drawing on dual-process cognitive models. The iterative variant extends the analogy
   to biological root growth — branching toward semantic "nutrients" and stopping in
   barren directions.
5. **Empirical analysis of the (n, τ) parameter space:** Systematic evaluation of how
  graph traversal depth and pruning threshold interact across batch and iterative
   strategies, including evidence for whether (n, τ) self-regulation holds.
6. **A systematic comparison** of six KG integration strategies for RAG (baseline,
  naive expansion, batch-pruned expansion, iteratively-pruned expansion, re-ranking,
   filtering) on multi-hop QA, using identical infrastructure and evaluation metrics.
7. **A lightweight, fully reproducible experimental setup** using entirely free and
  open-source tools (Ollama, Neo4j CE, FAISS, spaCy, HotpotQA), demonstrating that
   meaningful GraphRAG research is accessible without cloud compute or API budgets.

## 3.6 Limitations (to acknowledge in the dissertation)

Be upfront about these — professors respect honesty:

1. **Small scale**: 150 questions is sufficient for a Master's thesis comparison study
  but results may not generalize to production-scale systems.
2. **KG quality**: LLM-extracted triples from a small corpus will be incomplete.
  This is itself an interesting finding to discuss (relates to the 65.8% coverage
   limitation from the literature).
3. **Single LLM**: Using only Mistral 7B. Results may differ with larger models.
  Discuss as future work.
4. **CPU inference**: Slower, but demonstrates accessibility. Does not affect
  correctness metrics.
5. **Static pruning threshold**: While SP-GQE-i makes the expansion *depth* adaptive,
  τ itself is still fixed globally. An adaptive τ that adjusts per node based on
   local graph density or per query based on question complexity is proposed as
   future work.
6. **Embedding-only divergence**: Pruning uses cosine similarity in embedding space,
  which may miss nuanced semantic relationships. A learned scorer (like SubgraphRAG's
   MLP) could outperform — but at the cost of training data and simplicity.
7. **No comparison with Self-RAG**: Self-RAG requires fine-tuned models. Discussed in
  related work as an orthogonal (complementary) approach. Combining Self-RAG's
   adaptive retrieval with SP-GQE's pruned expansion is proposed as future work.

---

# APPENDIX A: Key Paper Reference List

## Must-Read Papers (directly relevant to your dissertation)

### Surveys

1. **Han et al. (2025)** "Retrieval-Augmented Generation with Graphs (GraphRAG)"
  arXiv:2501.00309 — *The* comprehensive survey. Use for framework and taxonomy.
2. **Peng et al. (2024)** "Graph Retrieval-Augmented Generation: A Survey"
  arXiv:2408.08921 — First survey, good for canonical definitions.

### Methods

1. **Asai et al. (2024)** "Self-RAG: Learning to Retrieve, Generate, and Critique
  through Self-Reflection" ICLR 2024 Oral. arXiv:2310.11511
   github.com/AkariAsai/self-rag — Professor specifically mentioned this.
2. **SubgraphRAG (2025)** "Simple is Effective" ICLR 2025. arXiv:2410.20724
  github.com/Graph-COM/SubgraphRAG — Lightweight graph retrieval, validates
   "simple works" thesis.
3. **HybGRAG (2025)** ACL 2025. arXiv:2412.16311 — Hybrid retrieval, closest
  to your approach in spirit.
4. **Practical GraphRAG (2025)** arXiv:2507.03226 — RRF fusion strategy,
  practical deployment focus.
5. **KG-RAG (2024)** arXiv:2405.12035 — Chain of Explorations traversal algorithm.
6. **G-Retriever (2024)** arXiv:2402.07630 — Steiner Tree subgraph retrieval.

### Evaluation and Comparison

1. **"RAG vs. GraphRAG: A Systematic Evaluation"** arXiv:2502.11371 — Direct
  predecessor to your comparison study. Cite heavily in motivation.
2. **RAGAS (2024)** arXiv:2309.15217. EACL 2024 demo.
  github.com/explodinggradients/ragas — Your evaluation framework.
3. **Microsoft GraphRAG Benchmarks** github.com/microsoft/graphrag-benchmarking-datasets

### Dataset

1. **HotpotQA (2018)** Yang et al. "HotpotQA: A Dataset for Diverse, Explainable
  Multi-Hop Question Answering" EMNLP 2018. hotpotqa.github.io

### Foundational

1. **Lewis et al. (2020)** "Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks" NeurIPS 2020. — The original RAG paper.

## Nice-to-Have Papers (for deeper literature review)

1. Zhu et al. (2025) arXiv:2504.10499 — Updated graph roles survey
2. Microsoft GraphRAG original blog post and repo (2024)
3. GraphRAG-Bench (2025) arXiv:2506.02404 — Domain-specific benchmark

---

# APPENDIX B: Quick-Start Setup Commands

```bash
# 1. Create project structure
mkdir -p dissertation/{src,data,results,notebooks}

# 2. Docker Compose for Neo4j (save as docker-compose.yml)
# See docker-compose.yml template below

# 3. Python environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
pip install langchain faiss-cpu spacy neo4j ragas datasets

# 4. Install spaCy model
python -m spacy download en_core_web_sm

# 5. Install Ollama (from ollama.com), then:
ollama pull mistral
ollama pull nomic-embed-text

# 6. Download HotpotQA dev set
# From: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/dissertation2026
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

---

# APPENDIX C: Professor Communication Template

When presenting this to your professor, frame it as:

**Subject: Research Proposal — Semantically-Pruned Graph Expansion for RAG**

> Professor,
>
> Following your suggestion to explore GraphRAG mechanisms, I have defined the
> following research direction:
>
> **Research Question**: How does semantically-pruned graph expansion affect
> retrieval quality in RAG systems, and what is the interaction between graph
> traversal depth and pruning threshold for multi-hop QA?
>
> **Method**: I propose a "recall-and-verify" approach: first expand entities via
> knowledge graph traversal (recall), then prune semantically divergent entities
> using embedding-based divergence from both natural language sub-queries and
> structured Cypher query results (verify). The structured results also enrich
> the final augmented query as compact verified facts. I compare this against four
> baselines — vanilla RAG, naive (unpruned) graph expansion, graph-filtered
> retrieval, and graph re-ranking — evaluated on a HotpotQA subset using RAGAS
> metrics and standard QA metrics (F1, EM).
>
> **Contribution**: (1) A training-free, dual-source semantic pruning and enrichment
> mechanism for graph-expanded RAG, (2) systematic evaluation of the depth × threshold
> parameter space, (3) task-type analysis on single-hop vs. multi-hop questions.
>
> **Tools**: Entirely open-source and local (Ollama, Neo4j, FAISS, spaCy).
>
> I would appreciate your feedback on this direction and would welcome the
> opportunity to discuss it at your convenience.
>
> Best regards

---

*Document generated: 2026-02-14*
*Last updated: 2026-02-14*