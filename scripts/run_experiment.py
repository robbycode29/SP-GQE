#!/usr/bin/env python3
"""Run HotpotQA experiments: baselines + SP-GQE (n, τ) grid + plots.

Stacks:
  plan — Neo4j graph + FAISS; LLM via Groq API if GROQ_API_KEY, else Ollama if reachable,
        else extractive fallback. Embeddings: MiniLM when using Groq; Ollama nomic-embed-text when using Ollama.
  local — sentence-transformers MiniLM, in-memory graph, extractive answers (fast debug)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sp_gqe.experiment.embedder import Embedder
from sp_gqe.experiment.hotpot_loader import (
    download_hotpot_dev,
    iter_chunks,
    load_hotpot,
    sample_questions,
)
from sp_gqe.experiment.literature import LITERATURE_ROWS
from sp_gqe.experiment.metrics import (
    f1_score,
    retrieval_precision_at_k,
    supporting_titles,
)
from sp_gqe.experiment.neo4j_graph import Neo4jQuestionGraph, connect_neo4j
from sp_gqe.experiment.groq_client import answer_with_groq, groq_available, groq_model
from sp_gqe.experiment.ollama_client import (
    OllamaEmbedder,
    answer_with_mistral,
    ollama_available,
)
from sp_gqe.experiment.pipelines import (
    PipelineName,
    answer_extractive,
    build_kg_for_example,
    load_spacy,
    run_pipeline,
)
from sp_gqe.experiment.plots import bar_comparison, fungi_heatmap
from sp_gqe.experiment.retrieval import FaissRetriever
from sp_gqe.settings import neo4j_config


def _safe_qid(example: dict) -> str:
    raw = str(example.get("_id", "unknown"))
    return "hq_" + re.sub(r"[^a-zA-Z0-9_]", "_", raw)[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="HotpotQA questions (default 25: fits several Groq runs/day; use 150 for full protocol)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument(
        "--stack",
        choices=("plan", "local"),
        default="plan",
        help="plan: Neo4j + FAISS + Groq (if GROQ_API_KEY) or Ollama or extractive fallback; local: MiniLM + RAM + extractive",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.data_dir / "hotpot_dev_distractor_v1.json"
    download_hotpot_dev(raw_path)
    all_items = load_hotpot(raw_path)
    sample = sample_questions(all_items, args.sample_size, args.seed)

    nlp = load_spacy()
    neo4j_driver = None

    plan_fallback: list[str] = []
    llm_backend = ""

    if args.stack == "plan":
        uri, user, pw = neo4j_config()
        neo4j_driver = connect_neo4j(uri, user, pw)

        if groq_available():
            embedder = Embedder()

            def answerer(q: str, ctxs: list[str]) -> str:
                return answer_with_groq(q, ctxs)

            llm_backend = f"groq:{groq_model()}"
            stack_desc = (
                "- **Stack (plan):** Groq API `"
                + groq_model()
                + "` generation (T=0), `all-MiniLM-L6-v2` embeddings, "
                "Neo4j Bolt co-occurrence graph per question, FAISS."
            )
        elif ollama_available():
            embedder = OllamaEmbedder()

            def answerer(q: str, ctxs: list[str]) -> str:
                return answer_with_mistral(q, ctxs)

            llm_backend = "ollama"
            stack_desc = (
                "- **Stack (plan):** Ollama `nomic-embed-text` embeddings, Ollama `mistral` answers (T=0), "
                "Neo4j Bolt co-occurrence graph per question, FAISS."
            )
        else:
            plan_fallback.append(
                "No GROQ_API_KEY and Ollama unreachable at OLLAMA_HOST — using MiniLM embeddings + extractive answers; "
                "set GROQ_API_KEY for hosted LLM or install Ollama with `mistral` and `nomic-embed-text`."
            )
            embedder = Embedder()
            llm_backend = "extractive_fallback"

            def answerer(q: str, ctxs: list[str]) -> str:
                return answer_extractive(embedder, q, ctxs)

            stack_desc = (
                "- **Stack (plan, partial):** Neo4j graph + FAISS; **fallback** embeddings/reader "
                "(see `plan_fallback` in JSON)."
            )

        hm_f1_title = "SP-GQE: mean answer F1 (n × τ) — Neo4j graph"
        hm_pk_title = "SP-GQE: mean retrieval P@k (n × τ) — Neo4j graph"
    else:
        embedder = Embedder()

        def answerer(q: str, ctxs: list[str]) -> str:
            return answer_extractive(embedder, q, ctxs)

        stack_desc = (
            "- **Stack (local):** `all-MiniLM-L6-v2`, in-memory co-occurrence KG, extractive answers."
        )
        hm_f1_title = "SP-GQE: mean answer F1 (n × τ) — extractive reader"
        hm_pk_title = "SP-GQE: mean retrieval P@k (n × τ) — title overlap with gold"
        llm_backend = "local_extractive"

    n_grid = [1, 2, 3]
    tau_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    heatmap_f1 = np.zeros((len(n_grid), len(tau_grid)))
    heatmap_pk = np.zeros((len(n_grid), len(tau_grid)))

    pipeline_f1: dict[str, list[float]] = {
        "V-RAG": [],
        "GQE-RAG(n=2)": [],
        "SP-GQE(n=2,τ=0.5)": [],
        "SP-GQE-i(n=3,τ=0.5)": [],
        "GR-RAG": [],
        "GF-RAG": [],
    }
    pipeline_pk: dict[str, list[float]] = {k: [] for k in pipeline_f1}

    try:
        for ex in tqdm(sample, desc="HotpotQA subset"):
            chunks = iter_chunks(ex)
            chunk_texts = [c[1] for c in chunks]
            gold = ex["answer"]
            supp = supporting_titles(ex)
            q = ex["question"]

            if args.stack == "plan":
                assert neo4j_driver is not None
                qid = _safe_qid(ex)
                kg = Neo4jQuestionGraph(neo4j_driver, qid)
                kg.load_from_example(nlp, ex)
            else:
                kg = build_kg_for_example(nlp, ex)

            retriever = FaissRetriever(embedder.dim)
            retriever.add(embedder.encode(chunk_texts), chunk_texts)

            def pf(name: PipelineName, **kw):
                pred, _st, ctxs = run_pipeline(
                    name,
                    nlp,
                    embedder,
                    q,
                    ex,
                    chunk_texts,
                    kg,
                    answerer=answerer,
                    top_k=args.top_k,
                    retriever=retriever,
                    **kw,
                )
                return pred, ctxs

            pred, ctxs = pf("V-RAG")
            pipeline_f1["V-RAG"].append(f1_score(pred, gold))
            pipeline_pk["V-RAG"].append(
                retrieval_precision_at_k(ctxs, supp, args.top_k)
            )

            pred, ctxs = pf("GQE-RAG", n_hops=2)
            pipeline_f1["GQE-RAG(n=2)"].append(f1_score(pred, gold))
            pipeline_pk["GQE-RAG(n=2)"].append(
                retrieval_precision_at_k(ctxs, supp, args.top_k)
            )

            pred, ctxs = pf("SP-GQE", n_hops=2, tau=0.5)
            pipeline_f1["SP-GQE(n=2,τ=0.5)"].append(f1_score(pred, gold))
            pipeline_pk["SP-GQE(n=2,τ=0.5)"].append(
                retrieval_precision_at_k(ctxs, supp, args.top_k)
            )

            pred, ctxs = pf("SP-GQE-i", n_hops=3, tau=0.5)
            pipeline_f1["SP-GQE-i(n=3,τ=0.5)"].append(f1_score(pred, gold))
            pipeline_pk["SP-GQE-i(n=3,τ=0.5)"].append(
                retrieval_precision_at_k(ctxs, supp, args.top_k)
            )

            pred, ctxs = pf("GR-RAG")
            pipeline_f1["GR-RAG"].append(f1_score(pred, gold))
            pipeline_pk["GR-RAG"].append(
                retrieval_precision_at_k(ctxs, supp, args.top_k)
            )

            pred, ctxs = pf("GF-RAG")
            pipeline_f1["GF-RAG"].append(f1_score(pred, gold))
            pipeline_pk["GF-RAG"].append(
                retrieval_precision_at_k(ctxs, supp, args.top_k)
            )

            for i, n in enumerate(n_grid):
                for j, tau in enumerate(tau_grid):
                    pred, _st, ctxs = run_pipeline(
                        "SP-GQE",
                        nlp,
                        embedder,
                        q,
                        ex,
                        chunk_texts,
                        kg,
                        answerer=answerer,
                        n_hops=n,
                        tau=tau,
                        top_k=args.top_k,
                        retriever=retriever,
                    )
                    heatmap_f1[i, j] += f1_score(pred, gold)
                    heatmap_pk[i, j] += retrieval_precision_at_k(
                        ctxs, supp, args.top_k
                    )

            if args.stack == "plan":
                kg.clear()
    finally:
        if neo4j_driver is not None:
            neo4j_driver.close()

    n_q = float(len(sample))
    heatmap = heatmap_f1 / n_q
    heatmap_p = heatmap_pk / n_q

    summary = {
        "stack": args.stack,
        "llm_backend": llm_backend,
        "plan_fallback": plan_fallback if args.stack == "plan" else [],
        "sample_size": len(sample),
        "seed": args.seed,
        "pipelines": {
            k: {
                "mean_f1": float(np.mean(v)),
                "std_f1": float(np.std(v)),
            }
            for k, v in pipeline_f1.items()
        },
        "retrieval_precision_at_k": {
            k: float(np.mean(v)) for k, v in pipeline_pk.items()
        },
        "heatmap": {
            "n_hops": n_grid,
            "tau": tau_grid,
            "mean_f1_grid": heatmap.tolist(),
            "mean_retrieval_p_at_k_grid": heatmap_p.tolist(),
        },
        "literature": LITERATURE_ROWS,
    }

    out_json = args.out_dir / "run_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fungi_heatmap(
        heatmap,
        n_grid,
        tau_grid,
        args.out_dir / "heatmap_fungi_n_tau.png",
        title=hm_f1_title,
        cbar_label="Mean answer F1",
    )
    fungi_heatmap(
        heatmap_p,
        n_grid,
        tau_grid,
        args.out_dir / "heatmap_fungi_n_tau_retrieval_p_at_k.png",
        title=hm_pk_title,
        cbar_label="Mean P@k",
    )

    labels = list(summary["pipelines"].keys())
    vals = [summary["pipelines"][k]["mean_f1"] for k in labels]
    bar_comparison(labels, vals, args.out_dir / "pipelines_bar_f1.png")

    md_lines = [
        f"# Experiment run (`--stack {args.stack}`)",
        "",
        f"- **Sample:** {len(sample)} HotpotQA distractor dev questions (seed {args.seed}).",
        stack_desc,
    ]
    if plan_fallback:
        md_lines.extend(["", "**Note:** " + " ".join(plan_fallback)])
    md_lines.extend(
        [
            "- **KG:** spaCy `en_core_web_sm` entities; **co-occurrence** edges (sentence-level). "
            "Dissertation LLM triple extraction can replace edge construction later.",
            "",
            "## Mean answer F1",
            "",
            "| Pipeline | Mean F1 | Std |",
            "|----------|---------|-----|",
        ]
    )
    for k, v in summary["pipelines"].items():
        md_lines.append(f"| {k} | {v['mean_f1']:.4f} | {v['std_f1']:.4f} |")

    md_lines.extend(["", "## Retrieval P@k (title overlap with supporting facts)", ""])
    for k, v in summary["retrieval_precision_at_k"].items():
        md_lines.append(f"- **{k}**: {v:.3f}")

    md_lines.extend(
        [
            "",
            "## SP-GQE heatmaps (n × τ)",
            "",
            "Answer F1:",
            "",
            "![f1](heatmap_fungi_n_tau.png)",
            "",
            "Retrieval P@k:",
            "",
            "![p_at_k](heatmap_fungi_n_tau_retrieval_p_at_k.png)",
            "",
            "## Positioning vs published RAG / GraphRAG systems",
            "",
            "| System | Reference | Notes |",
            "|--------|-----------|-------|",
        ]
    )
    for row in LITERATURE_ROWS:
        md_lines.append(
            f"| {row['system']} | {row['ref']} | {row['notes']} |"
        )
    md_lines.append("")
    md_path = args.out_dir / "EXPERIMENT_REPORT.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {out_json}, {md_path}, plots in {args.out_dir}")


if __name__ == "__main__":
    main()
