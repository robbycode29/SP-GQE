# Heatmap rerun estimate — all 562 questions (seed-by-seed, Groq free tier)

This answers: **How long / how many UTC days** to reproduce every archived seed with `**--stack plan`**, **single seed**, **heatmap ON** (omit `--no-heatmap`), matching current `results/daily_runs` sample sizes — and whether that is **more rigorous** for (*n*, *τ*).

## What the code does (cost model)

From `scripts/run_experiment.py`:

- **Every question:** **6** baseline reader calls (V-RAG, GQE-RAG, SP-GQE at (2,0.5), SP-GQE-i, GR-RAG, GF-RAG).
- **If heatmap:** **+15** extra **SP-GQE** runs (grid **n ∈ {1,2,3}** × **τ ∈ {0.3,…,0.7}**).
- **Total:** **21 Groq reader calls per question**.

**Heatmaps require `len(seeds)==1`.** You must run **23 separate commands** (one per archived seed), matching `--seed` and `--sample-size` from `AGGREGATED_SUMMARY.json` → `seeds_used`.

## Archived seeds and sample sizes


| Seeds     | Questions per seed | Subtotal Q |
| --------- | ------------------ | ---------- |
| 42–47     | 25                 | 6×25 = 150 |
| 48        | 15                 | 15         |
| 49–56     | 25                 | 8×25 = 200 |
| 57        | 22                 | 22         |
| 58–64     | 25                 | 7×25 = 175 |
| **Total** | **23 runs**        | **562**    |


## Groq free-tier constraints (defaults in `groq_client.py`)


| Limit   | Default                                            | Implication for this job                             |
| ------- | -------------------------------------------------- | ---------------------------------------------------- |
| **TPD** | 500 000 tokens (−2 000 margin → **~498 000** safe) | **Primary calendar bottleneck**                      |
| **RPD** | 14 400 requests                                    | Total requests **562×21 = 11 802** → **under** RPD ✓ |
| **RPM** | 30                                                 | ~**2 s** minimum spacing between calls               |
| **TPM** | 6 000 tokens / rolling minute                      | Extra **wait** when bursts exceed window             |


## Token estimate (order of magnitude)

Empirical runs often land near **~300 recorded tokens per Groq call** (varies with passage length).

- **Per question:** 21 × ~300 ≈ **6 300** tokens.  
- **562 questions:** 562 × 6 300 ≈ **3.54×10⁶** tokens total.

**UTC days from TPD alone**

- 3 540 000 / 498 000 ≈ **7.1** → plan for **8 UTC days** if usage matches ~300 tok/call.  
- If calls average **~450** tokens: ~**11** UTC days.

You **cannot** safely finish **all 562** heatmap questions in **one** UTC day on **500k TPD** without paid quota or a cheaper reader path.

## Wall-clock per seed (rough)

Dominant effects: **21 calls/question**, **RPM**, **TPM**, network latency.

- Observed **without** heatmap: often **~15–25 s/question** (6 calls).  
- **Scaling:** ~~21/6 ≈ **3.5×** calls → **~~1–1.5 min/question** is a reasonable planning range after TPM waits.

**Per seed (approximate active runtime)**


| Sample size | Wall time (order of magnitude) |
| ----------- | ------------------------------ |
| 15          | ~15–25 min                     |
| 22          | ~22–35 min                     |
| 25          | ~25–40 min                     |


**All 23 seeds sequentially:** roughly **~10–14 hours** of *active* generation time, **spread across multiple UTC days** because of **TPD**, not because each seed takes a full day.

**Practical schedule:** aim for **~2–3 “n=25” seeds per UTC day** (~~485k tokens/day at ~308 tok/call), plus lighter days for seeds **48** and **57** mixed with fewer n=25 runs — **~~7–10 UTC days** total is a realistic band.

## Command pattern (repeat per seed)

```powershell
cd SP-GQE
.\.venv\Scripts\python.exe scripts\run_experiment.py --stack plan --seed <SEED> --sample-size <N>
# Do NOT pass --no-heatmap; do NOT pass multi-seed --seeds
```

Use `<SEED>` / `<N>` from the table above. Each run overwrites `results/run_summary.json`; persist outputs under `results/daily_runs/` as usual (`_persist_daily_run`).

## Does this make the (*n*, *τ*) comparison “more rigorous”?

**Partially — with caveats.**


| Stronger                                                                                                                                                                     | Still limited                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Each heatmap cell is a mean over **the real seed sample** (15–25 questions), not a single tiny pilot.                                                                        | The grid still evaluates **only SP-GQE** variants; **V-RAG / GR-RAG / …** stay **fixed** configs.                                               |
| Repeating for **23 seeds** gives **23 independent heatmap matrices** → you can **element-wise average** grids across seeds for a **stabler** sensitivity plot than one seed. | The codebase does **not** auto-aggregate heatmaps across seeds; you would **post-process** JSON outputs or extend `aggregate_daily_runs.py`.    |
| You report **both** mean **F1** and mean **P@k** grids (`heatmap_fungi_*.png`).                                                                                              | This is **not** full Bayesian hyperparameter optimisation or dev-wide tuning; it is **controlled sensitivity** on your **562-instance** corpus. |


**Bottom line for the professor:** rerunning heatmaps seed-by-seed on the **same** questions as the aggregate is **materially more evidence** than a **single-seed** heatmap, but **rigour** still means clearly stating: (*i*) **SP-GQE-only** grid, (*ii*) **post-hoc** sensitivity unless (*n*,*τ*) were fixed **before** any peek (your protocol fixed **n=2, τ=0.5** for H1), (*iii**) pooling across seeds requires an explicit aggregation rule.

---

*Figures use defaults in `run_experiment.py`: `n_grid = [1,2,3]`, `tau_grid = [0.3,0.4,0.5,0.6,0.7]`.*