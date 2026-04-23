# SP-GQE evaluation protocol (HotpotQA distractor dev)

## Pre-registered claim (hypothesis)

**H1:** SP-GQE improves **answer quality** (token F1) over **V-RAG** on questions where **multi-hop linking of evidence** is central — operationalised as HotpotQA `bridge` items — while holding the **same dense index**, **same reader**, and **same graph-construction family** (per-question RDF co-occurrence graph) fixed.

**Null expectation:** On `comparison` items (typically two-hop comparison between two entities) dense retrieval often suffices; SP-GQE is not expected to beat V-RAG.

## Two-branch SP-GQE (as implemented)

For every question `q`:

1. **Graph construction.** An RDF graph is built from the 10 paragraphs provided in the HotpotQA distractor setting. Triples:
   - `<e> a spg:Entity` for every spaCy-NER entity;
   - `<e> rdfs:label "normalised name"`;
   - `<a> spg:coOccurs <b>` for every pair of entities that co-occur in the same sentence (both directions stored, so the relation is symmetric).
2. **Branch 1 — structural n-hop traversal (SPARQL 1.1).** Seed entities = spaCy-NER entities of the question. A SPARQL query with a bounded property path (`{ ?s spg:coOccurs ?t } UNION { ?s spg:coOccurs/spg:coOccurs ?t } UNION ...`) returns all entities reachable within ≤ n hops. Call this set `A`.
3. **Branch 2 — keyword-driven semantic SPARQL.** Keywords are extracted from the question's noun chunks (lowercased, deduplicated, stop-word filtered, length ≥ 3). A SPARQL query with a `CONTAINS(LCASE(STR(?label)), …)` filter over `rdfs:label` returns entities whose surface form matches any keyword. Call this set `B`.
4. **Fusion + pruning.** Candidates are `(A ∪ B) \ seeds`. Each candidate is scored by the **max cosine similarity** between its MiniLM embedding and the embeddings of the **reunion** `{question} ∪ noun_chunks`. Candidates with score ≥ τ are kept; seeds are always kept.
5. **Augmented dense retrieval.** The FAISS top-k is queried with `question + " Graph context: " + describe_entities(kept) + " Verified probes: " + probes`.
6. **Reader.** Groq `llama-3.1-8b-instant` (T = 0) is called with the retrieved chunks and returns a short answer.

Ablations (same FAISS index, same reader, same graph):

- **V-RAG** — no graph, plain FAISS on the question.
- **GQE-RAG(n=2)** — Branch 1 only, no pruning (raw n-hop neighbours appended to the question).
- **SP-GQE(n=2, τ=0.5)** — full two-branch SPARQL + cosine pruning (primary).
- **SP-GQE-i(n=3, τ=0.5)** — iterative single-branch variant (Branch 1 only, pruning applied at each hop).
- **GR-RAG, GF-RAG** — lexical entity-reranking / filtering controls.

## Metrics

### Answer quality (primary / secondary)

| Role | Metric | Definition |
|------|--------|------------|
| **Primary** | Mean **token F1** | Normalised token overlap between prediction and gold short answer (HotpotQA-style). |
| **Secondary** | Mean **exact match (EM)** | 0/1 normalised string equality on the answer. |
| **Secondary** | **Supporting-title recall @k** | Fraction of gold supporting-paragraph titles that appear in the top-k retrieved chunks. |
| **Diagnostic** | **Retrieval P@k** | Fraction of top-k chunks whose title matches a supporting fact. |

### Graph-query validity (ablation requested by reviewer)

Per question we record the entities returned by each SPARQL branch and the set kept after τ pruning, and we compute **precision / recall against the set of supporting entities** — i.e. spaCy-NER entities extracted from the gold supporting paragraphs. Four evaluation points are reported per question:

| Stage | Set | What it measures |
|-------|-----|------------------|
| Branch 1 (n-hop) | `A` | Quality of the structural SPARQL expansion alone. |
| Branch 2 (keyword) | `B` | Quality of the semantic SPARQL lookup alone. |
| Union | `A ∪ B` | What enters the similarity-prune stage. |
| Kept at τ | `kept` | What is actually fed into the augmented FAISS query. |

This tests two separable properties of SP-GQE: whether the **graph query itself** returns useful entities (branch precision/recall), and whether the **semantic pruner** improves precision without destroying recall.

## Statistical design

- **Stratified sampling:** `sample_questions` draws half `bridge`, half `comparison` per seed.
- **Multi-seed, multi-day runs:** each RNG seed produces a self-contained per-seed file under `results/daily_runs/<date>__seed<seed>__n<N>.json`. Accumulating files from multiple days is safe and cumulative.
- **Aggregation:** `scripts/aggregate_daily_runs.py` reads every file in `results/daily_runs/` and reports **mean ± 95 % CI** across seed-level means (t-interval) for every metric, and **bootstrap 95 % CI** for the paired ΔF1 (SP-GQE − V-RAG) pooled across all pairs. Re-running the aggregator after adding more seeds is idempotent.
- **Subset analysis:** metrics are reported separately for `bridge` and `comparison`.

## Multi-day budget plan (Groq free tier)

- Reader: `llama-3.1-8b-instant`; TPD = 500 000; empirically ≈ 8 000 tokens per question across the six pipelines.
- **Recommendation:** seed pairs of 20 questions per day (≈ 320 000 TPD per pair, well under the cap). Target: 8 seeds × 20 Q = 160 unique question instances over 4 days.
- Always run with `--no-heatmap` after the first single-seed run; heatmaps multiply LLM calls by `|n_grid| × |τ_grid|`.

## Comparability

Dataset, split and chunking follow HotpotQA distractor dev; answer scoring uses `metrics.f1_score` / `exact_match`. External KBs are not used; the per-question RDF graph is built exclusively from the 10 provided paragraphs.
