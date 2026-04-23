# Addressing reviewer (professor) feedback

This document maps each point of the professor's feedback to the concrete
change made in the repository. Everything below is implemented; running
`python scripts/run_experiment.py --seeds 42,43 --no-heatmap` produces a
self-contained per-seed file under `results/daily_runs/` each day, and
`python scripts/aggregate_daily_runs.py` aggregates all days into a final
report.

## 1. Correctness of the description given to the reviewer

Three things in the description were imprecise at the time they were given;
**they are all now true of the code**:

| Claim to the reviewer | Status at the time | Current status |
|-----------------------|--------------------|----------------|
| "SPARQL query over the graph" | The code used **Cypher** over **Neo4j**. | The graph is now an in-memory **RDF graph** (`rdflib.Graph`) and both branches are **SPARQL 1.1** queries. |
| "Two branches: one n-hop traversal, one semantic query from keywords" | Only one branch (n-hop) existed; noun chunks were used only for pruning. | Branch 1 and Branch 2 are both SPARQL and run before pruning, as described (`SP-GQE/src/sp_gqe/experiment/rdf_graph.py`, `SP-GQE/src/sp_gqe/experiment/pipelines.py` → `if name == "SP-GQE"`). |
| "The union of the SPARQL result with the natural query is used as the cosine reference for pruning" | The pruning probes were only noun chunks. | The **reunion** is now `{question} ∪ noun_chunks` and every candidate is scored by max cosine against it. |
| "n noduri" | *n* is actually the **number of hops**, not nodes. | Unchanged in substance — the code still exposes *n* as hops; the correction here is terminological for the thesis prose. |
| "Ollama limitat la 40–60 pe Groq" | The 40–60 was the sample size; the reader was Groq `llama-3.1-8b-instant`, not Ollama. | Clarified: reader = Groq; Ollama is only a fallback when `GROQ_API_KEY` is absent. |

## 2. Ablation study of the SPARQL-query validity

For every question we compute, **per question**, the precision and recall of
each graph-query stage against the **gold supporting entities** (spaCy-NER
entities of the HotpotQA supporting paragraphs):

- Branch 1 (SPARQL n-hop traversal) alone → precision/recall vs. supporting entities
- Branch 2 (SPARQL keyword lookup) alone → precision/recall
- Union `A ∪ B` → precision/recall
- Kept after τ = 0.5 pruning → precision/recall

This is stored per-question in `graph_query_log[*].validity` in every daily
run file, and aggregated across seeds by
`scripts/aggregate_daily_runs.py` into the table under *"Graph-query validity"*
in `results/AGGREGATED_REPORT.md`. It directly answers the reviewer's ask:
*"as evalua si calitatea / relevanta query-ului SPARQL"*.

Reading the table:

- **Branch 1 precision vs. Branch 2 precision** — is one branch mostly
  delivering noise? If yes, the two-branch design is not justified for the
  current graph.
- **Union recall − individual-branch recall** — how complementary are the two
  SPARQL queries? A meaningful gain means Branch 2 finds supporting entities
  that the structural traversal misses (and vice-versa).
- **Kept precision − Union precision** — how much the semantic pruner
  improves precision.
- **Union recall − Kept recall** — the cost paid by the pruner in dropped
  gold entities.

## 3. SPARQL query samples (qualitative study) + where they live in code

### Where the queries are built

- `SP-GQE/src/sp_gqe/experiment/rdf_graph.py`:
  - `RdfQuestionGraph.build_n_hop_sparql(seeds, n)` — Branch 1 query text.
  - `RdfQuestionGraph.n_hop_neighbors(seeds, n)` — executes it.
  - `RdfQuestionGraph.build_keyword_sparql(noun_chunks)` — Branch 2 query text.
  - `RdfQuestionGraph.keyword_entities(noun_chunks)` — executes it.
- `SP-GQE/src/sp_gqe/experiment/pipelines.py` → the block guarded by
  `if name == "SP-GQE"` is where both branches are called during evaluation.
- `SP-GQE/scripts/run_experiment.py::_sp_gqe_trace` calls the same `build_*`
  helpers purely for **logging**, so the exact SPARQL text used for every
  question is recorded without an extra LLM call.

### How qualitative samples are produced

For each seed, the first 10 questions of the seed are written to
`results/daily_runs/<date>__seed<N>__n<N>__queries.md`. Per question the file
shows:

- the two SPARQL queries literally, in ` ```sparql ` fenced blocks;
- the entities each branch returns;
- the union, the top-10 similarity scores, and the kept set after τ;
- the gold supporting entities and the four precision/recall numbers;
- a checklist for the human reviewer (Branch 1 well-formed, Branch 2
  well-formed, pruner OK, construction gap, …).

This is the "simple valid way" requested: read 10 SPARQL queries per day,
tick the boxes, aggregate the labels across 4–6 days for a manual quality
table in the thesis.

## 4. Larger sample over a multi-day run

- `--sample-size 20`, `--seeds 42,43` fits one day under Groq's TPD (≈ 320 k
  tokens, cap = 500 k).
- Each seed writes a self-contained JSON under `results/daily_runs/`, so
  re-running on a subsequent day with different seeds is additive.
- **Recommended 4-day plan** (8 seeds, 160 question instances, 80 per stratum):
  - Day 1: `--seeds 42,43 --sample-size 20 --no-heatmap`
  - Day 2: `--seeds 44,45 --sample-size 20 --no-heatmap`
  - Day 3: `--seeds 46,47 --sample-size 20 --no-heatmap`
  - Day 4: `--seeds 48,49 --sample-size 20 --no-heatmap`
  - After each day: `python scripts/aggregate_daily_runs.py`.
- Between days, reset `GROQ_TOKENS_USED_TODAY_INITIAL=0` in `.env` and delete
  `data/.groq_quota_state.json` (see `results/GROQ_MULTI_SEED_NOTE.md`).

## 5. Heatmap ×  KG overlay

`scripts/plot_kg_overlay.py` renders, side-by-side, the per-question KG
(seed nodes in red, kept nodes in blue, pruned nodes in grey) and the
`n × τ` mean-F1 heatmap from a single-seed run. It is intentionally stub-
quality — to be invoked only **after** the multi-day data is in, to cherry-
pick two or three visually clean examples for the thesis appendix:

```bash
python scripts/plot_kg_overlay.py \
    --sample-json results/daily_runs/2026-04-23__seed42__n20.json \
    --heatmap-json results/run_summary.json \
    --question-idx 3
```

## 6. How each day's run is saved and later aggregated

### Where every seed run goes

- `results/daily_runs/<YYYY-MM-DD>__seed<N>__n<N>.json` — full machine-
  readable payload: per-pipeline metrics, paired ΔF1, graph-query log,
  graph-query validity, literature rows, backend.
- `results/daily_runs/<YYYY-MM-DD>__seed<N>__n<N>__queries.md` — first-10
  SPARQL-query samples with ground-truth precision/recall and a manual
  review checklist.

These files are **never overwritten** across days (the filename is
date-stamped).

### Aggregator (`scripts/aggregate_daily_runs.py`)

Reads every `results/daily_runs/*.json`, then produces:

- `results/AGGREGATED_SUMMARY.json` — every aggregated number, including
  seed-level means, t-CIs, and the pooled bootstrap ΔF1 CI;
- `results/AGGREGATED_REPORT.md` — the table version of the above, plus the
  graph-query-validity ablation table.

Both files are regenerated from scratch on every aggregator run, so the
aggregator is idempotent and safe to run after every day.
