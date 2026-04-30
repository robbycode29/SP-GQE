# Debug visual run (2026-04-23)

## Commands

```powershell
cd SP-GQE
# Small run: local stack (MiniLM + extractive), single seed → heatmaps enabled
.\.venv\Scripts\python.exe scripts\run_experiment.py --stack local --seed 42 --sample-size 3

# Per-question figures (repeat for --question-idx 0 1 2)
.\.venv\Scripts\python.exe scripts\plot_kg_overlay.py `
  --sample-json results\daily_runs\2026-04-23__seed42__n3.json `
  --heatmap-json results\run_summary.json `
  --question-idx 0
```

## Outputs


| Artifact                                     | Path                                                    |
| -------------------------------------------- | ------------------------------------------------------- |
| Full run JSON (SPARQL + edges + validity)    | `results/daily_runs/2026-04-23__seed42__n3.json`        |
| Markdown SPARQL samples (first 10 questions) | `results/daily_runs/2026-04-23__seed42__n3__queries.md` |
| Run summary + heatmap grids                  | `results/run_summary.json`                              |
| Side-by-side: RDF subgraph + n×τ heatmap     | `results/debug/kg_sidebyside_q{0,1,2}.png`              |
| Overlay: heatmap with RDF subgraph inset     | `results/debug/kg_heatmap_inset_q{0,1,2}.png`           |


The left panel / inset uses **real** `spg:coOccurs` edges among entities in the spotlight set (seeds ∪ union), similar to a GraphDB “Explore” view: `spg:Entity` nodes with `rdfs:label` and undirected co-occurrence links.