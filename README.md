# SP-GQE

**Semantically-Pruned Graph Query Expansion** for Retrieval-Augmented Generation — a recall-and-verify pipeline over a knowledge graph with embedding-based pruning.

Dissertation materials and the full research plan live under `[dissertation/](dissertation/)`.

## Repository layout


| Path                     | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `src/sp_gqe/`            | Python package (pip install -e .)                      |
| `config/experiment.yaml` | Experiment matrix, metrics, dataset sampling           |
| `data/`                  | HotpotQA and derived artifacts (gitignored when large) |
| `results/`               | Run outputs and tables                                 |
| `notebooks/`             | Exploratory work                                       |
| `dissertation/`          | Plan, preview, notes, professor message                |


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

- **Protocol and hypotheses**: `dissertation/DISSERTATION_PLAN.md` (Section 3).
- **Runnable config**: `config/experiment.yaml` — pipelines (V-RAG … GF-RAG), run IDs E1–E6, fixed parameters, metrics list.
- **Implementation** will add data prep, KG construction, FAISS index, and pipeline scripts under `src/sp_gqe/` following that plan.

## License

Add a license file if you publish the repo publicly.