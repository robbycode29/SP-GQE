#!/usr/bin/env python3
"""Run HotpotQA experiments: baselines + optional SP-GQE (n, τ) grid.

See config/EXPERIMENT_PROTOCOL.md for the pre-registered hypothesis and metrics.

Stacks:
  plan — RDFLib SPARQL graph + FAISS; Groq if GROQ_API_KEY, else Ollama, else extractive.
  local — same RDF graph + FAISS; MiniLM + extractive reader (no API keys).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sp_gqe.experiment.embedder import Embedder, cosine_sim_matrix
from sp_gqe.experiment.groq_client import answer_with_groq, groq_available, groq_model
from sp_gqe.experiment.hotpot_loader import (
    download_hotpot_dev,
    iter_chunks,
    load_hotpot,
    sample_questions,
)
from sp_gqe.experiment.kg import norm_entity
from sp_gqe.experiment.literature import LITERATURE_ROWS
from sp_gqe.experiment.metrics import (
    exact_match,
    f1_score,
    retrieval_precision_at_k,
    supporting_title_recall_at_k,
    supporting_titles,
)
from sp_gqe.experiment.nlp_utils import extract_entities, noun_chunks
from sp_gqe.experiment.rdf_graph import RdfQuestionGraph
from sp_gqe.experiment.ollama_client import (
    OllamaEmbedder,
    answer_with_mistral,
    ollama_available,
)
from sp_gqe.experiment.pipelines import (
    PipelineName,
    answer_extractive,
    load_spacy,
    run_pipeline,
)
from sp_gqe.experiment.plots import bar_comparison, fungi_heatmap
from sp_gqe.experiment.retrieval import FaissRetriever

PROTOCOL_PATH = REPO / "config" / "EXPERIMENT_PROTOCOL.md"

PRIMARY_METRIC = "mean_token_f1"
SECONDARY_METRICS = ("answer_exact_match", "supporting_title_recall_at_k")

PIPELINE_KEYS = [
    "V-RAG",
    "GQE-RAG(n=2)",
    "SP-GQE(n=2,τ=0.5)",
    "SP-GQE-i(n=3,τ=0.5)",
    "GR-RAG",
    "GF-RAG",
]


def _safe_qid(example: dict) -> str:
    raw = str(example.get("_id", "unknown"))
    return "hq_" + re.sub(r"[^a-zA-Z0-9_]", "_", raw)[:200]


def _supporting_entities(nlp: Any, example: dict) -> set[str]:
    """Entities (normalised) that appear in gold supporting paragraphs or titles.

    Used only for the graph-query validity metric (not for answer scoring)."""
    sf = example.get("supporting_facts", []) or []
    titles = {t[0] for t in sf if t}
    ents: set[str] = set()
    for para in example.get("context", []) or []:
        if not para or len(para) < 2:
            continue
        title, sents = para[0], para[1]
        if title in titles:
            for e in extract_entities(nlp, title):
                ents.add(norm_entity(e))
            for sent in sents:
                for e in extract_entities(nlp, sent):
                    ents.add(norm_entity(e))
    return {e for e in ents if e}


def _sp_gqe_trace(
    nlp: Any,
    embedder: Any,
    kg: Any,
    question: str,
    *,
    n_hops: int = 2,
    tau: float = 0.5,
) -> dict[str, Any]:
    """Reproduce SP-GQE two-branch internals for per-question graph-query inspection.

    Branch 1 — structural n-hop traversal (SPARQL property paths).
    Branch 2 — keyword-driven semantic SPARQL over rdfs:label.
    Fusion  — (A ∪ B) \\ seeds, cosine-pruned against the reunion {question} ∪ probes.
    """
    q_ents = {norm_entity(e) for e in extract_entities(nlp, question)}
    subqs = noun_chunks(nlp, question) or [question[:80]]
    probes = subqs[:12]
    seeds = q_ents or {norm_entity(question[:40])}

    branch1 = kg.n_hop_neighbors(seeds, n_hops)
    branch2 = kg.keyword_entities(probes) if hasattr(kg, "keyword_entities") else set()
    union = branch1 | branch2

    cand_list = [e for e in union if e not in seeds]
    sims: list[list[Any]] = []
    kept: list[str] = []
    if cand_list:
        reunion = [question] + probes
        pv = embedder.encode(reunion)
        cv = embedder.encode(cand_list)
        s_mat = cosine_sim_matrix(cv, pv)
        max_per = s_mat.max(axis=1)
        for i, ent in enumerate(cand_list):
            ms = float(max_per[i])
            sims.append([ent, round(ms, 4)])
            if ms >= tau:
                kept.append(ent)
    kept_set = set(kept) | seeds

    sparql_nhop = (
        kg.build_n_hop_sparql(seeds, n_hops)
        if hasattr(kg, "build_n_hop_sparql")
        else ""
    )
    sparql_kw = (
        kg.build_keyword_sparql(probes)
        if hasattr(kg, "build_keyword_sparql")
        else ""
    )

    spotlight = set(seeds) | union
    co_edges: list[list[str]] = []
    uri_map: dict[str, str] = {}
    if hasattr(kg, "cooccurrence_edges_among"):
        co_edges = [
            [a, b] for a, b in kg.cooccurrence_edges_among(spotlight)  # type: ignore[attr-defined]
        ]
    if hasattr(kg, "entity_uri_map"):
        uri_map = dict(kg.entity_uri_map(spotlight))  # type: ignore[attr-defined]

    return {
        "seeds": sorted(seeds),
        "probes": probes,
        "n_hops": n_hops,
        "tau": tau,
        "branch1_sparql_nhop": sparql_nhop,
        "branch1_returned": sorted(branch1),
        "branch1_returned_n": int(len(branch1)),
        "branch2_sparql_keyword": sparql_kw,
        "branch2_returned": sorted(branch2),
        "branch2_returned_n": int(len(branch2)),
        "union_returned": sorted(union),
        "returned_n": int(len(union)),
        "per_candidate_similarity": sims,
        "kept_tau": sorted(kept_set),
        "kept_n": int(len(kept_set)),
        "subgraph_spotlight_nodes": sorted(spotlight),
        "cooccurrence_edges": co_edges,
        "entity_uri_by_label": uri_map,
    }


def _graph_validity(
    branch1: set[str],
    branch2: set[str],
    union: set[str],
    kept: set[str],
    supporting: set[str],
) -> dict[str, float]:
    def _p(s: set[str]) -> float:
        return (len(s & supporting) / len(s)) if s else 0.0

    def _r(s: set[str]) -> float:
        return (len(s & supporting) / len(supporting)) if supporting else 0.0

    return {
        "supporting_entities_n": float(len(supporting)),
        "branch1_n": float(len(branch1)),
        "branch2_n": float(len(branch2)),
        "union_n": float(len(union)),
        "kept_n": float(len(kept)),
        "graph_precision_branch1": _p(branch1),
        "graph_recall_branch1": _r(branch1),
        "graph_precision_branch2": _p(branch2),
        "graph_recall_branch2": _r(branch2),
        "graph_precision_union": _p(union),
        "graph_recall_union": _r(union),
        "graph_precision_kept": _p(kept),
        "graph_recall_kept": _r(kept),
    }


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _mean_ci_t(values: list[float]) -> tuple[float, tuple[float, float]]:
    """95% CI for the mean using t-distribution (for seed-level means)."""
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    m = float(np.mean(a))
    if n == 1:
        return m, (m, m)
    sem = stats.sem(a, ddof=1)
    h = sem * stats.t.ppf(0.975, df=n - 1)
    return m, (m - h, m + h)


def _bootstrap_ci_mean_diff(
    deltas: list[float],
    seed: int = 42,
    n_boot: int = 8000,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap 95% CI for mean of paired differences."""
    a = np.asarray(deltas, dtype=np.float64)
    if len(a) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(a)
    for i in range(n_boot):
        means[i] = rng.choice(a, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(a)), (float(lo), float(hi))


def run_experiment_once(
    *,
    sample: list[dict[str, Any]],
    args: argparse.Namespace,
    nlp: Any,
    embedder: Any,
    answerer: Callable[[str, list[str]], str],
    heatmap: bool,
) -> dict[str, Any]:
    """One stratified sample (one seed). Returns summary fragment + raw paired rows."""

    n_grid = [1, 2, 3]
    tau_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    heatmap_f1 = np.zeros((len(n_grid), len(tau_grid)))
    heatmap_pk = np.zeros((len(n_grid), len(tau_grid)))

    pipeline_f1: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    pipeline_em: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    pipeline_suprec: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    pipeline_pk: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}

    paired_bridge: list[dict[str, float]] = []
    paired_comp: list[dict[str, float]] = []

    graph_query_log: list[dict[str, Any]] = []
    gv_b1_p: list[float] = []
    gv_b1_r: list[float] = []
    gv_b2_p: list[float] = []
    gv_b2_r: list[float] = []
    gv_union_p: list[float] = []
    gv_union_r: list[float] = []
    gv_kept_p: list[float] = []
    gv_kept_r: list[float] = []

    for ex in tqdm(sample, desc="HotpotQA subset"):
            chunks = iter_chunks(ex)
            chunk_texts = [c[1] for c in chunks]
            gold = ex["answer"]
            supp = supporting_titles(ex)
            q = ex["question"]
            qtype = str(ex.get("type", "unknown"))

            # Same RDF + SPARQL backend for plan and local (local uses extractive reader).
            kg = RdfQuestionGraph()
            kg.load_from_example(nlp, ex)

            retriever = FaissRetriever(embedder.dim)
            retriever.add(embedder.encode(chunk_texts), chunk_texts)

            def pf(name: PipelineName, **kw: Any):
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

            results: dict[str, tuple[str, list[str]]] = {}
            results["V-RAG"] = pf("V-RAG")
            results["GQE-RAG(n=2)"] = pf("GQE-RAG", n_hops=2)
            results["SP-GQE(n=2,τ=0.5)"] = pf("SP-GQE", n_hops=2, tau=0.5)
            results["SP-GQE-i(n=3,τ=0.5)"] = pf("SP-GQE-i", n_hops=3, tau=0.5)
            results["GR-RAG"] = pf("GR-RAG")
            results["GF-RAG"] = pf("GF-RAG")

            for pk, (pred, ctxs) in results.items():
                pipeline_f1[pk].append(f1_score(pred, gold))
                pipeline_em[pk].append(exact_match(pred, gold))
                pipeline_suprec[pk].append(
                    supporting_title_recall_at_k(ctxs, supp, args.top_k)
                )
                pipeline_pk[pk].append(
                    retrieval_precision_at_k(ctxs, supp, args.top_k)
                )

            f1_vr = f1_score(results["V-RAG"][0], gold)
            f1_sp = f1_score(results["SP-GQE(n=2,τ=0.5)"][0], gold)
            row = {"v_rag_f1": f1_vr, "sp_gqe_f1": f1_sp, "delta_f1": f1_sp - f1_vr}
            if qtype == "bridge":
                paired_bridge.append(row)
            elif qtype == "comparison":
                paired_comp.append(row)

            trace = _sp_gqe_trace(nlp, embedder, kg, q, n_hops=2, tau=0.5)
            supp_ents = _supporting_entities(nlp, ex)
            b1 = set(trace["branch1_returned"])
            b2 = set(trace["branch2_returned"])
            union_set = set(trace["union_returned"])
            kept_set = set(trace["kept_tau"])
            validity = _graph_validity(b1, b2, union_set, kept_set, supp_ents)
            gv_b1_p.append(validity["graph_precision_branch1"])
            gv_b1_r.append(validity["graph_recall_branch1"])
            gv_b2_p.append(validity["graph_precision_branch2"])
            gv_b2_r.append(validity["graph_recall_branch2"])
            gv_union_p.append(validity["graph_precision_union"])
            gv_union_r.append(validity["graph_recall_union"])
            gv_kept_p.append(validity["graph_precision_kept"])
            gv_kept_r.append(validity["graph_recall_kept"])
            graph_query_log.append(
                {
                    "qid": str(ex.get("_id", "")),
                    "type": qtype,
                    "question": q,
                    "gold_answer": gold,
                    "sp_gqe_f1": f1_sp,
                    "v_rag_f1": f1_vr,
                    **trace,
                    "supporting_titles": sorted(supp),
                    "supporting_entities": sorted(supp_ents),
                    "validity": validity,
                }
            )

            if heatmap:
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

            # RdfQuestionGraph is per-question; no explicit cleanup needed (GC).

    n_q = float(len(sample))
    per_pipeline: dict[str, Any] = {}
    for k in PIPELINE_KEYS:
        per_pipeline[k] = {
            "mean_f1": float(np.mean(pipeline_f1[k])),
            "std_f1": float(np.std(pipeline_f1[k])),
            "mean_em": float(np.mean(pipeline_em[k])),
            "mean_supporting_title_recall_at_k": float(np.mean(pipeline_suprec[k])),
            "mean_retrieval_p_at_k": float(np.mean(pipeline_pk[k])),
        }

    out: dict[str, Any] = {
        "sample_size": len(sample),
        "pipelines": per_pipeline,
        "paired_deltas": {
            "bridge": paired_bridge,
            "comparison": paired_comp,
        },
        "graph_query_log": graph_query_log,
        "graph_query_validity": {
            "mean_graph_precision_branch1_nhop": (
                float(np.mean(gv_b1_p)) if gv_b1_p else 0.0
            ),
            "mean_graph_recall_branch1_nhop": (
                float(np.mean(gv_b1_r)) if gv_b1_r else 0.0
            ),
            "mean_graph_precision_branch2_keyword": (
                float(np.mean(gv_b2_p)) if gv_b2_p else 0.0
            ),
            "mean_graph_recall_branch2_keyword": (
                float(np.mean(gv_b2_r)) if gv_b2_r else 0.0
            ),
            "mean_graph_precision_union": (
                float(np.mean(gv_union_p)) if gv_union_p else 0.0
            ),
            "mean_graph_recall_union": (
                float(np.mean(gv_union_r)) if gv_union_r else 0.0
            ),
            "mean_graph_precision_kept": (
                float(np.mean(gv_kept_p)) if gv_kept_p else 0.0
            ),
            "mean_graph_recall_kept": (
                float(np.mean(gv_kept_r)) if gv_kept_r else 0.0
            ),
            "n_questions": len(graph_query_log),
        },
    }

    if heatmap and n_q > 0:
        out["heatmap"] = {
            "n_hops": n_grid,
            "tau": tau_grid,
            "mean_f1_grid": (heatmap_f1 / n_q).tolist(),
            "mean_retrieval_p_at_k_grid": (heatmap_pk / n_q).tolist(),
        }

    return out


def _persist_daily_run(
    daily_dir: Path,
    *,
    run: dict[str, Any],
    seed: int,
    sample_size: int,
    stack: str,
    llm_backend: str,
    top_k: int,
) -> None:
    """Save a self-contained JSON + qualitative-review Markdown for one seed.

    Multi-day aggregation reads every file in daily_dir/*.json afterwards.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stem = f"{date}__seed{seed}__n{sample_size}"
    payload = {
        "date_utc": date,
        "seed": seed,
        "sample_size": sample_size,
        "stack": stack,
        "llm_backend": llm_backend,
        "top_k": top_k,
        **run,
    }
    (daily_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log = run.get("graph_query_log", []) or []
    if not log:
        return
    md = [
        f"# SPARQL query samples — {date}, seed {seed} (n={sample_size})",
        "",
        "Qualitative review of the two-branch SP-GQE(n=2, τ=0.5) pipeline.",
        "For each question we show:",
        "",
        "- the **Branch 1** SPARQL query (structural n-hop traversal from seed entities) "
        "and the entities it returns;",
        "- the **Branch 2** SPARQL query (keyword-driven lookup over `rdfs:label`) and "
        "the entities it returns;",
        "- the **union** of branches, the per-candidate cosine similarity to the reunion "
        "`{question} ∪ noun_chunks`, and the set **kept at τ = 0.5**;",
        "- the **gold supporting entities** extracted from the HotpotQA supporting facts "
        "and the per-branch precision/recall (graph-query validity ablation).",
        "",
    ]
    for i, entry in enumerate(log[:10], 1):
        sims_sorted = sorted(
            entry.get("per_candidate_similarity", []), key=lambda x: -x[1]
        )
        v = entry.get("validity", {})
        b1 = entry.get("branch1_returned", [])
        b2 = entry.get("branch2_returned", [])
        md += [
            f"## {i}. [{entry['type']}] {entry['question']}",
            "",
            f"- **qid:** `{entry['qid']}`",
            f"- **gold answer:** {entry['gold_answer']}",
            f"- **SP-GQE F1 / V-RAG F1:** {entry['sp_gqe_f1']:.3f} / {entry['v_rag_f1']:.3f}",
            f"- **seed entities (spaCy NER, normalised):** {entry['seeds']}",
            f"- **noun-chunk probes:** {entry['probes']}",
            "",
            "### Branch 1 — structural n-hop (SPARQL)",
            "",
            "```sparql",
            entry.get("branch1_sparql_nhop", ""),
            "```",
            "",
            f"- **returned (n={entry.get('branch1_returned_n', len(b1))}):** "
            + (str(b1[:25]) + ("..." if len(b1) > 25 else "")),
            "",
            "### Branch 2 — keyword / semantic (SPARQL)",
            "",
            "```sparql",
            entry.get("branch2_sparql_keyword", ""),
            "```",
            "",
            f"- **returned (n={entry.get('branch2_returned_n', len(b2))}):** "
            + (str(b2[:25]) + ("..." if len(b2) > 25 else "")),
            "",
            "### Fusion and pruning",
            "",
            f"- **union |A ∪ B| = {entry['returned_n']}**",
            f"- **kept after τ = 0.5 (n = {entry['kept_n']}):** {entry['kept_tau']}",
            "- **top-10 candidate similarities (max cosine vs. reunion `{question} ∪ probes`):**",
            "",
        ]
        for ent, s in sims_sorted[:10]:
            md.append(f"  - `{ent}` — {s:.3f}")
        md += [
            "",
            "### Ground truth & validity",
            "",
            f"- **supporting titles:** {entry['supporting_titles']}",
            f"- **supporting entities (spaCy NER of gold paragraphs):** "
            + str(entry['supporting_entities'][:25])
            + ("..." if len(entry['supporting_entities']) > 25 else ""),
            "",
            "| Stage | Precision | Recall |",
            "|-------|-----------|--------|",
            f"| Branch 1 (n-hop) | {v.get('graph_precision_branch1', 0.0):.3f} | "
            f"{v.get('graph_recall_branch1', 0.0):.3f} |",
            f"| Branch 2 (keyword) | {v.get('graph_precision_branch2', 0.0):.3f} | "
            f"{v.get('graph_recall_branch2', 0.0):.3f} |",
            f"| Union | {v.get('graph_precision_union', 0.0):.3f} | "
            f"{v.get('graph_recall_union', 0.0):.3f} |",
            f"| Kept after τ | {v.get('graph_precision_kept', 0.0):.3f} | "
            f"{v.get('graph_recall_kept', 0.0):.3f} |",
            "",
            "**Manual review (tick as appropriate):**",
            "",
            "- [ ] Branch 1 SPARQL is well-formed and returns ≥ 1 supporting entity",
            "- [ ] Branch 2 SPARQL is well-formed and returns ≥ 1 supporting entity",
            "- [ ] τ pruning removed clearly off-topic candidates",
            "- [ ] τ pruning did not drop a gold supporting entity that was present in the union",
            "- [ ] Any supporting entity was missing from both branches (graph-construction gap)",
            "",
            "---",
            "",
        ]
    (daily_dir / f"{stem}__queries.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Questions per seed (stratified bridge/comparison)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Single-seed run if --seeds omitted",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (e.g. 42,43,44). Enables multi-seed aggregation.",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument(
        "--stack",
        choices=("plan", "local"),
        default="plan",
        help=(
            "plan: RDFLib SPARQL graph + FAISS + Groq/Ollama/fallback reader; "
            "local: MiniLM + RDFLib SPARQL co-occurrence graph + extractive reader"
        ),
    )
    ap.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip n×τ grid (saves many LLM calls). Recommended under tight Groq TPD.",
    )
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = [args.seed]

    heatmap_on = (not args.no_heatmap) and len(seeds) == 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.data_dir / "hotpot_dev_distractor_v1.json"
    download_hotpot_dev(raw_path)
    all_items = load_hotpot(raw_path)

    nlp = load_spacy()
    plan_fallback: list[str] = []
    llm_backend = ""
    embedder: Any = None
    answerer: Callable[[str, list[str]], str]
    stack_desc = ""

    if args.stack == "plan":
        if groq_available():
            embedder = Embedder()

            def _ans(q: str, ctxs: list[str]) -> str:
                return answer_with_groq(q, ctxs)

            answerer = _ans
            llm_backend = f"groq:{groq_model()}"
            stack_desc = (
                "- **Stack (plan):** Groq API `"
                + groq_model()
                + "` generation (T=0), `all-MiniLM-L6-v2` embeddings, "
                "RDFLib in-memory per-question RDF graph queried via SPARQL 1.1, FAISS."
            )
        elif ollama_available():
            embedder = OllamaEmbedder()

            def _ans2(q: str, ctxs: list[str]) -> str:
                return answer_with_mistral(q, ctxs)

            answerer = _ans2
            llm_backend = "ollama"
            stack_desc = (
                "- **Stack (plan):** Ollama `nomic-embed-text` embeddings, Ollama `mistral` answers (T=0), "
                "RDFLib in-memory per-question RDF graph queried via SPARQL 1.1, FAISS."
            )
        else:
            plan_fallback.append(
                "No GROQ_API_KEY and Ollama unreachable — MiniLM + extractive answers."
            )
            embedder = Embedder()
            llm_backend = "extractive_fallback"

            def _ans3(q: str, ctxs: list[str]) -> str:
                return answer_extractive(embedder, q, ctxs)

            answerer = _ans3
            stack_desc = (
                "- **Stack (plan, partial):** Neo4j graph + FAISS; extractive reader fallback."
            )
    else:
        embedder = Embedder()

        def _ans4(q: str, ctxs: list[str]) -> str:
            return answer_extractive(embedder, q, ctxs)

        answerer = _ans4
        stack_desc = (
            "- **Stack (local):** `all-MiniLM-L6-v2`, RDFLib per-question RDF + SPARQL 1.1 graph, "
            "extractive answers."
        )
        llm_backend = "local_extractive"

    assert embedder is not None

    hm_f1_title = "SP-GQE: mean answer F1 (n × τ) — RDF / SPARQL co-occurrence graph"
    hm_pk_title = "SP-GQE: mean retrieval P@k (n × τ) — RDF / SPARQL co-occurrence graph"

    daily_dir = args.out_dir / "daily_runs"
    daily_dir.mkdir(parents=True, exist_ok=True)

    per_seed_results: list[dict[str, Any]] = []
    for sd in seeds:
        sample = sample_questions(all_items, args.sample_size, sd)
        one = run_experiment_once(
            sample=sample,
            args=args,
            nlp=nlp,
            embedder=embedder,
            answerer=answerer,
            heatmap=heatmap_on,
        )
        one["seed"] = sd
        per_seed_results.append(one)
        _persist_daily_run(
            daily_dir,
            run=one,
            seed=sd,
            sample_size=args.sample_size,
            stack=args.stack,
            llm_backend=llm_backend,
            top_k=args.top_k,
        )

    # Aggregate seed-level means for primary/secondary metrics
    seed_means_f1: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    seed_means_em: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    seed_means_sup: dict[str, list[float]] = {k: [] for k in PIPELINE_KEYS}
    for run in per_seed_results:
        for k in PIPELINE_KEYS:
            seed_means_f1[k].append(run["pipelines"][k]["mean_f1"])
            seed_means_em[k].append(run["pipelines"][k]["mean_em"])
            seed_means_sup[k].append(run["pipelines"][k]["mean_supporting_title_recall_at_k"])

    aggregated: dict[str, Any] = {}
    for k in PIPELINE_KEYS:
        m_f1, ci_f1 = _mean_ci_t(seed_means_f1[k])
        m_em, ci_em = _mean_ci_t(seed_means_em[k])
        m_sr, ci_sr = _mean_ci_t(seed_means_sup[k])
        aggregated[k] = {
            "primary_metric": PRIMARY_METRIC,
            "mean_f1": m_f1,
            "ci95_f1_seed_runs": list(ci_f1),
            "mean_em": _clip01(m_em),
            "ci95_em_seed_runs": [_clip01(ci_em[0]), _clip01(ci_em[1])],
            "mean_supporting_title_recall_at_k": _clip01(m_sr),
            "ci95_supporting_title_recall_seed_runs": [
                _clip01(ci_sr[0]),
                _clip01(ci_sr[1]),
            ],
            "per_seed_mean_f1": seed_means_f1[k],
        }

    # Pool paired deltas across seeds for bootstrap CI (H1: bridge subset)
    bridge_deltas: list[float] = []
    comp_deltas: list[float] = []
    for run in per_seed_results:
        for row in run["paired_deltas"]["bridge"]:
            bridge_deltas.append(row["delta_f1"])
        for row in run["paired_deltas"]["comparison"]:
            comp_deltas.append(row["delta_f1"])

    bd_mean, bd_ci = _bootstrap_ci_mean_diff(bridge_deltas)
    cd_mean, cd_ci = _bootstrap_ci_mean_diff(comp_deltas)

    hypothesis_test = {
        "claim": (
            "SP-GQE(n=2,τ=0.5) improves mean token F1 vs V-RAG on bridge (multi-hop) "
            "questions; comparison subset is secondary."
        ),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
        "paired_mean_difference_SP_GQE_minus_V_RAG": {
            "bridge": {"mean_delta_f1": bd_mean, "bootstrap_ci95": list(bd_ci), "n_pairs": len(bridge_deltas)},
            "comparison": {"mean_delta_f1": cd_mean, "bootstrap_ci95": list(cd_ci), "n_pairs": len(comp_deltas)},
        },
    }

    summary: dict[str, Any] = {
        "protocol": str(PROTOCOL_PATH.relative_to(REPO)).replace("\\", "/"),
        "hypothesis": hypothesis_test["claim"],
        "stack": args.stack,
        "llm_backend": llm_backend,
        "plan_fallback": plan_fallback if args.stack == "plan" else [],
        "seeds": seeds,
        "sample_size_per_seed": args.sample_size,
        "total_question_instances": args.sample_size * len(seeds),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
        "aggregated_across_seeds": aggregated,
        "per_seed": per_seed_results,
        "hypothesis_test": hypothesis_test,
        "heatmap_included": heatmap_on,
        "literature": LITERATURE_ROWS,
    }

    if heatmap_on and per_seed_results and "heatmap" in per_seed_results[0]:
        summary["heatmap"] = per_seed_results[0]["heatmap"]

    out_json = args.out_dir / "run_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if heatmap_on and per_seed_results and "heatmap" in per_seed_results[0]:
        hg = per_seed_results[0]["heatmap"]
        n_grid = hg["n_hops"]
        tau_grid = hg["tau"]
        fungi_heatmap(
            np.array(hg["mean_f1_grid"]),
            n_grid,
            tau_grid,
            args.out_dir / "heatmap_fungi_n_tau.png",
            title=hm_f1_title,
            cbar_label="Mean answer F1",
        )
        fungi_heatmap(
            np.array(hg["mean_retrieval_p_at_k_grid"]),
            n_grid,
            tau_grid,
            args.out_dir / "heatmap_fungi_n_tau_retrieval_p_at_k.png",
            title=hm_pk_title,
            cbar_label="Mean P@k",
        )

    labels = PIPELINE_KEYS
    vals = [aggregated[k]["mean_f1"] for k in labels]
    bar_comparison(labels, vals, args.out_dir / "pipelines_bar_f1.png")

    md_lines = [
        "# Experiment run (`--stack " + args.stack + "`)",
        "",
        f"- **Protocol:** `{PROTOCOL_PATH.relative_to(REPO)}`",
        f"- **Hypothesis:** {hypothesis_test['claim']}",
        f"- **Primary metric:** {PRIMARY_METRIC}; **secondary:** {', '.join(SECONDARY_METRICS)}",
        "",
        f"- **Seeds:** {seeds} ({len(seeds)} runs × {args.sample_size} questions = {summary['total_question_instances']} instances).",
        stack_desc,
    ]
    if plan_fallback:
        md_lines.extend(["", "**Note:** " + " ".join(plan_fallback)])
    md_lines.extend(
        [
            "",
            "## Aggregated across seeds (mean ± 95% CI on seed-level means)",
            "",
            "| Pipeline | Mean F1 | 95% CI | Mean EM | 95% CI | Mean sup. title recall@k | 95% CI |",
            "|----------|---------|--------|---------|--------|---------------------------|--------|",
        ]
    )
    for k in PIPELINE_KEYS:
        a = aggregated[k]
        md_lines.append(
            f"| {k} | {a['mean_f1']:.4f} | [{a['ci95_f1_seed_runs'][0]:.4f}, {a['ci95_f1_seed_runs'][1]:.4f}] | "
            f"{a['mean_em']:.4f} | [{a['ci95_em_seed_runs'][0]:.4f}, {a['ci95_em_seed_runs'][1]:.4f}] | "
            f"{a['mean_supporting_title_recall_at_k']:.4f} | "
            f"[{a['ci95_supporting_title_recall_seed_runs'][0]:.4f}, {a['ci95_supporting_title_recall_seed_runs'][1]:.4f}] |"
        )

    h = hypothesis_test["paired_mean_difference_SP_GQE_minus_V_RAG"]
    md_lines.extend(
        [
            "",
            "## Mechanism test (paired SP-GQE − V-RAG on token F1)",
            "",
            f"- **Bridge (H1):** mean Δ = {h['bridge']['mean_delta_f1']:.4f}, "
            f"bootstrap 95% CI [{h['bridge']['bootstrap_ci95'][0]:.4f}, {h['bridge']['bootstrap_ci95'][1]:.4f}], "
            f"n = {h['bridge']['n_pairs']}",
            f"- **Comparison:** mean Δ = {h['comparison']['mean_delta_f1']:.4f}, "
            f"bootstrap 95% CI [{h['comparison']['bootstrap_ci95'][0]:.4f}, {h['comparison']['bootstrap_ci95'][1]:.4f}], "
            f"n = {h['comparison']['n_pairs']}",
        ]
    )

    if heatmap_on:
        md_lines.extend(
            [
                "",
                "## SP-GQE heatmaps (n × τ) — seed " + str(seeds[0]) + " only",
                "",
                "![f1](heatmap_fungi_n_tau.png)",
                "",
                "![p_at_k](heatmap_fungi_n_tau_retrieval_p_at_k.png)",
            ]
        )
    else:
        md_lines.append("")
        md_lines.append("*Heatmaps skipped (`--no-heatmap` or multi-seed).*")

    md_lines.extend(
        [
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
