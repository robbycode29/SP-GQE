"""RAG pipelines: V-RAG, GQE-RAG, SP-GQE, SP-GQE-i."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import spacy
from spacy.language import Language

from sp_gqe.experiment.embedder import Embedder, cosine_sim_matrix
from sp_gqe.experiment.kg import CooccurrenceKG, norm_entity
from sp_gqe.experiment.neo4j_graph import Neo4jQuestionGraph
from sp_gqe.experiment.nlp_utils import extract_entities, noun_chunks
from sp_gqe.experiment.rdf_graph import RdfQuestionGraph
from sp_gqe.experiment.retrieval import FaissRetriever

GraphBackend = Union[CooccurrenceKG, Neo4jQuestionGraph, RdfQuestionGraph]
EmbedLike = Union[Embedder, object]


@dataclass
class RunStats:
    expansion_raw: int
    expansion_after_prune: int
    latency_ms: float


def load_spacy() -> Language:
    return spacy.load("en_core_web_sm")


def build_kg_for_example(nlp: Language, example: dict) -> CooccurrenceKG:
    kg = CooccurrenceKG()
    for para in example["context"]:
        sents = para[1]
        for sent in sents:
            ents = extract_entities(nlp, sent)
            if len(ents) >= 2:
                kg.add_clique(ents)
            elif len(ents) == 1:
                t_ents = extract_entities(nlp, para[0])
                for te in t_ents:
                    kg.add_clique([ents[0], te])
    return kg


def describe_entities(entities: list[str], max_items: int = 24) -> str:
    entities = entities[:max_items]
    return "; ".join(entities)


def answer_extractive(
    embedder: EmbedLike,
    question: str,
    contexts: list[str],
) -> str:
    if not contexts:
        return ""
    qv = embedder.encode([question])[0]
    best_sent = ""
    best_score = -1.0
    for ctx in contexts:
        sents = re.split(r"(?<=[.!?])\s+", ctx)
        for s in sents:
            s = s.strip()
            if len(s) < 8:
                continue
            sv = embedder.encode([s])[0]
            sc = float(np.dot(qv, sv))
            if sc > best_score:
                best_score = sc
                best_sent = s
    return best_sent[:300]


PipelineName = Literal["V-RAG", "GQE-RAG", "SP-GQE", "SP-GQE-i", "GR-RAG", "GF-RAG"]


def run_pipeline(
    name: PipelineName,
    nlp: Language,
    embedder: EmbedLike,
    question: str,
    example: dict,
    chunk_texts: list[str],
    kg: GraphBackend,
    *,
    answerer: Callable[[str, list[str]], str],
    n_hops: int = 2,
    tau: float = 0.5,
    top_k: int = 5,
    retriever: Optional[FaissRetriever] = None,
) -> tuple[str, RunStats, list[str]]:
    t0 = time.perf_counter()
    q_ents = set(norm_entity(e) for e in extract_entities(nlp, question))
    subqs = noun_chunks(nlp, question)
    if not subqs:
        subqs = [question[:80]]

    if retriever is None:
        retriever = FaissRetriever(embedder.dim)  # type: ignore[attr-defined]
        retriever.add(embedder.encode(chunk_texts), chunk_texts)

    seeds = q_ents or {norm_entity(question[:40])}

    def retrieve_with_query(qtext: str) -> list[str]:
        qv = embedder.encode([qtext])[0]
        hits = retriever.search(qv, top_k)
        return [h[0] for h in hits]

    expansion_raw = 0
    expansion_after = 0

    if name == "V-RAG":
        ctxs = retrieve_with_query(question)
        pred = answerer(question, ctxs)
        ms = RunStats(0, 0, (time.perf_counter() - t0) * 1000)
        return pred, ms, ctxs

    if name == "GQE-RAG":
        expanded = kg.n_hop_neighbors(seeds, n_hops)
        expansion_raw = len(expanded)
        expansion_after = expansion_raw
        aug = f"{question} Graph context: {describe_entities(list(expanded))}"
        ctxs = retrieve_with_query(aug)
        pred = answerer(question, ctxs)
        ms = RunStats(expansion_raw, expansion_after, (time.perf_counter() - t0) * 1000)
        return pred, ms, ctxs

    if name == "SP-GQE":
        # Branch 1 — structural: SPARQL n-hop traversal from seed entities.
        branch1 = kg.n_hop_neighbors(seeds, n_hops)
        # Branch 2 — semantic: SPARQL keyword lookup driven by question noun chunks.
        probes = subqs[:12]
        if hasattr(kg, "keyword_entities"):
            branch2 = kg.keyword_entities(probes)
        else:
            branch2 = set()
        # Fusion: (A ∪ B) minus seeds, then cosine-prune vs. reunion probes.
        union = branch1 | branch2
        expansion_raw = max(1, len(union))
        cand_list = [e for e in union if e not in seeds]
        reunion = [question] + probes
        if cand_list:
            pv = embedder.encode(reunion)
            cv = embedder.encode(cand_list)
            sims = cosine_sim_matrix(cv, pv).max(axis=1)
            kept = [cand_list[i] for i in range(len(cand_list)) if sims[i] >= tau]
        else:
            kept = []
        kept_set = set(kept) | seeds
        expansion_after = len(kept_set)
        aug = (
            f"{question} Graph context: {describe_entities(list(kept_set))} "
            f"Verified probes: {'; '.join(probes[:6])}"
        )
        ctxs = retrieve_with_query(aug)
        pred = answerer(question, ctxs)
        ms = RunStats(expansion_raw, expansion_after, (time.perf_counter() - t0) * 1000)
        return pred, ms, ctxs

    if name == "SP-GQE-i":
        frontier = set(seeds)
        acc = set(seeds)
        pv = embedder.encode(subqs[:12])
        for _ in range(n_hops):
            nbrs = kg.one_hop(frontier) - acc
            expansion_raw += len(nbrs)
            if not nbrs:
                break
            cand_list = list(nbrs)
            cv = embedder.encode(cand_list)
            sims = cosine_sim_matrix(cv, pv).max(axis=1)
            survivors = {cand_list[i] for i in range(len(cand_list)) if sims[i] >= tau}
            acc |= survivors
            frontier = survivors
            if not frontier:
                break
        expansion_after = len(acc)
        aug = f"{question} Graph context: {describe_entities(list(acc))}"
        ctxs = retrieve_with_query(aug)
        pred = answerer(question, ctxs)
        ms = RunStats(expansion_raw, expansion_after, (time.perf_counter() - t0) * 1000)
        return pred, ms, ctxs

    if name == "GR-RAG":
        ctxs_base = retrieve_with_query(question)
        if not q_ents:
            pred = answerer(question, ctxs_base)
            return pred, RunStats(0, 0, (time.perf_counter() - t0) * 1000), ctxs_base
        scores = []
        for c in ctxs_base:
            low = c.lower()
            bonus = sum(1 for e in q_ents if e in low)
            scores.append((bonus, c))
        scores.sort(key=lambda x: -x[0])
        ctxs_rr = [c for _, c in scores]
        pred = answerer(question, ctxs_rr)
        return pred, RunStats(0, 0, (time.perf_counter() - t0) * 1000), ctxs_rr

    if name == "GF-RAG":
        filt = []
        for ch in chunk_texts:
            low = ch.lower()
            if any(e in low for e in q_ents):
                filt.append(ch)
        pool = filt if filt else chunk_texts
        vecs2 = embedder.encode(pool)
        r2 = FaissRetriever(embedder.dim)  # type: ignore[attr-defined]
        r2.add(vecs2, pool)
        qv = embedder.encode([question])[0]
        hits = r2.search(qv, top_k)
        ctxs = [h[0] for h in hits]
        pred = answerer(question, ctxs)
        return pred, RunStats(0, 0, (time.perf_counter() - t0) * 1000), ctxs

    raise ValueError(f"unknown pipeline {name}")
