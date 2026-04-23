# Groq multi-seed run

**Completed:** `python scripts/run_experiment.py --stack plan --seeds 42,43,44 --sample-size 20 --no-heatmap`

- **Outputs:** `run_summary.json`, `EXPERIMENT_REPORT.md`, `pipelines_bar_f1.png`
- **Quota file:** `data/.groq_quota_state.json` (tokens/requests for this machine; plus `GROQ_TOKENS_USED_TODAY_INITIAL` from `.env` for dashboard alignment)

Before the next Groq-heavy run, set **`GROQ_TOKENS_USED_TODAY_INITIAL`** in `.env` to the Groq console “tokens used today” value.

**Interrupted run (older):** If TPD blocked mid-run, reset initial + optional `data/.groq_quota_state.json` on a new UTC day and re-run the same command.
