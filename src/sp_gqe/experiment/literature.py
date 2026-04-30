"""Published reference points for positioning (not reproduced in this repo).

Figures are approximate / headline results from papers; always cite the original work.
"""

from __future__ import annotations

# Rows: method, venue/year, dataset/setting, metric, notes
LITERATURE_ROWS: list[dict[str, str]] = [
    {
        "system": "Vanilla RAG (DPR-style dense retriever + reader)",
        "ref": "Lewis et al., NeurIPS 2020; follow-up retrieval stacks",
        "setting": "Open-domain QA / multi-hop settings vary",
        "hotpot_f1": "~58–62 (representative strong dense-pipeline range on dev)*",
        "notes": "Depends on reader; HotpotQA distractor is harder than full-wiki.",
    },
    {
        "system": "Microsoft GraphRAG (community summaries)",
        "ref": "Edge et al., 2024; msft graphrag",
        "setting": "Corpus-specific graph; not identical to HotpotQA distractor",
        "hotpot_f1": "N/A (different task mix; strong on global queries)",
        "notes": "Heavy offline indexing; not directly comparable sample-for-sample.",
    },
    {
        "system": "HybGRAG",
        "ref": "arXiv:2412.16311 (ACL 2025)",
        "setting": "STaRK + hybrid textual/relational retrieval",
        "hotpot_f1": "Reports large gains on STaRK Hit@1; HotpotQA not primary table",
        "notes": "Demonstrates hybrid retrieval > single-modality.",
    },
    {
        "system": "SubgraphRAG",
        "ref": "ICLR 2025; arXiv:2410.20724",
        "setting": "KG subgraph retrieval with lightweight scorer",
        "hotpot_f1": "Strong on KG-QA benchmarks in paper; architecture-specific",
        "notes": "Trained scorer vs our training-free pruning.",
    },
    {
        "system": "RAG vs GraphRAG systematic study",
        "ref": "arXiv:2502.11371 (Feb 2025)",
        "setting": "Text benchmarks incl. multi-hop QA",
        "hotpot_f1": "Neither RAG nor GraphRAG universally dominates",
        "notes": "Motivates hybrid / query-aware graph use (aligned with SP-GQE).",
    },
    {
        "system": "SP-GQE (this work)",
        "ref": "Repository experiments",
        "setting": "HotpotQA distractor dev subset + co-occurrence KG + extractive answer",
        "hotpot_f1": "See results/run_summary.json — relative trends > absolute SOTA",
        "notes": "Prototype stack: meant for ablations vs baselines under identical code.",
    },
]
