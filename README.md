# SP-GQE

**Semantically-Pruned Graph Query Expansion** for Retrieval-Augmented Generation — a recall-and-verify pipeline over a knowledge graph with embedding-based pruning.

Dissertation materials and the full research plan live under `[dissertation/](dissertation/)`.

## Repository layout


| Path                     | Purpose                                                        |
| ------------------------ | -------------------------------------------------------------- |
| `src/sp_gqe/`            | Python package (pip install -e .)                              |
| `src/sp_gqe/experiment/` | Hotpot loader, Neo4j graph, Ollama client, pipelines, metrics  |
| `config/experiment.yaml` | Experiment matrix, metrics, dataset sampling                   |
| `data/`                  | HotpotQA and derived artifacts (gitignored when large)         |
| `results/`               | Run outputs and tables (`AGGREGATED_REPORT.md`, `daily_runs/`) |
| `deliverables/`          | Generated empirical report (Markdown / DOCX)                   |
| `notebooks/`             | Exploratory work                                               |


## Prerequisites

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/) (for Neo4j)
- [Ollama](https://ollama.com) (local LLM + embeddings)

## Quick start

1. **Clone and enter the repo** (if you have not already).
2. **Environment file**
  Copy `.env.example` to `.env` and adjust if needed.
3. **Python venv and dependencies**
  ```powershell
   .\scripts\init_project.ps1
  ```
4. **Ollama models** (matches `config/experiment.yaml` defaults)
  ```text
   ollama pull mistral
   ollama pull nomic-embed-text
  ```
5. **Neo4j**
  ```text
   docker compose up -d
  ```
   Browser: [http://localhost:7474](http://localhost:7474) — user `neo4j`, password `dissertation2026` (see `.env.example`).
6. **Verify Python can load settings**
  ```powershell
   .\.venv\Scripts\Activate.ps1
   python -c "from sp_gqe.settings import experiment_config; print(experiment_config()['dataset']['name'])"
  ```

## Experimental setup

- **Protocol and hypotheses**: `config/EXPERIMENT_PROTOCOL.md` (pre-registered metrics & H1); broader plan: `dissertation/DISSERTATION_PLAN.md`.
- **Runnable config**: `config/experiment.yaml` — pipelines, run IDs, metrics (legacy naming may differ from the HotpotQA script path below).
- **HotpotQA distractor evaluation** (`scripts/run_experiment.py`):
  - `--stack plan`: per-question **RDFLib / SPARQL** co-occurrence graph, **FAISS** + `all-MiniLM-L6-v2`, reader **Groq** `llama-3.1-8b-instant` if `GROQ_API_KEY` is set, else Ollama **mistral**, else extractive fallback.
  - `--stack local`: same graph + MiniLM + **extractive** reader (fast, no API keys).
  - Example:
    ```powershell
    Set-Location SP-GQE   # repo root
    .\.venv\Scripts\python.exe scripts\run_experiment.py --stack plan --seed 42 --sample-size 25
    ```
  - Multi-seed / quota: use `--no-heatmap` after your first single-seed heatmap run (see `results/GROQ_MULTI_SEED_NOTE.md`).
- **Neo4j / Docker** in this repo supports other graph workflows; the HotpotQA experiment above does **not** require Neo4j.

## Experimental results (research aggregate)

Aggregated from all `results/daily_runs/*.json` (excluding `archive/`). Regenerate after new runs:

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_daily_runs.py
```

Outputs: `results/AGGREGATED_REPORT.md`, `results/AGGREGATED_SUMMARY.json`.

**Snapshot (last aggregated):** **562** question instances, **23** seeds (HotpotQA distractor dev subset; stratified bridge/comparison per seed). Reader: Groq `llama-3.1-8b-instant` where noted in per-seed files.

### Mean answer & retrieval metrics (95% CI on seed-level means)


| Pipeline              | Mean F1    | 95% CI F1        | Mean EM |
| --------------------- | ---------- | ---------------- | ------- |
| GQE-RAG(n=2)          | 0.5760     | [0.5381, 0.6139] | 0.4636  |
| GR-RAG                | 0.5727     | [0.5361, 0.6092] | 0.4607  |
| V-RAG                 | 0.5633     | [0.5276, 0.5991] | 0.4520  |
| SP-GQE-i(n=3,τ=0.5)   | 0.5520     | [0.5212, 0.5828] | 0.4560  |
| GF-RAG                | 0.5512     | [0.5213, 0.5811] | 0.4403  |
| **SP-GQE(n=2,τ=0.5)** | **0.5489** | [0.5149, 0.5828] | 0.4463  |


Full table (sup-title recall@k, P@k): see `AGGREGATED_REPORT.md`.

### Paired Δ token F1 — SP-GQE(n=2,τ=0.5) minus V-RAG (bootstrap 95% CI)


| Subset     | Mean Δ  | 95% CI            | n pairs |
| ---------- | ------- | ----------------- | ------- |
| bridge     | −0.0185 | [−0.0473, 0.0093] | 270     |
| comparison | −0.0096 | [−0.0448, 0.0253] | 292     |


Both intervals include zero: **no statistically sharp advantage** for SP-GQE over V-RAG on this sample; interpret together with dissertation discussion.

### Graph-query validity (entity sets vs gold supporting NER), pooled


| Stage                     | Mean precision | Mean recall |
| ------------------------- | -------------- | ----------- |
| Branch 1 (n-hop SPARQL)   | 0.3460         | 0.5959      |
| Branch 2 (keyword SPARQL) | 0.3793         | 0.1698      |
| Union (pre-τ)             | 0.3155         | 0.6343      |
| Kept after τ=0.5          | 0.4545         | 0.2192      |


Long-form write-up: `deliverables/SP_GQE_Empirical_Report.md`.

## License

Add a license file if you publish the repo publicly.