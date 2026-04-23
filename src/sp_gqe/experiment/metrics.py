"""HotpotQA-style F1 / EM."""

from __future__ import annotations

import re
from collections import Counter


def normalize_answer(s: str) -> str:

    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    if not p or not g:
        return float(p == g)
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def retrieval_precision_at_k(
    retrieved_chunks: list[str],
    supporting_titles: set[str],
    k: int,
) -> float:
    """Fraction of top-k chunks whose title matches a supporting fact title."""
    hits = 0
    for chunk in retrieved_chunks[:k]:
        title = chunk.split(".", 1)[0].strip().lower()
        if any(st.lower() in title or title in st.lower() for st in supporting_titles):
            hits += 1
    return hits / k if k else 0.0


def supporting_titles(example: dict) -> set[str]:
    return {t[0] for t in example.get("supporting_facts", [])}


def chunk_title(chunk: str) -> str:
    return chunk.split(".", 1)[0].strip().lower()


def supporting_title_recall_at_k(
    retrieved_chunks: list[str],
    supporting_titles: set[str],
    k: int,
) -> float:
    """Fraction of gold supporting paragraph titles that appear in top-k retrieved chunks."""
    if not supporting_titles:
        return 0.0
    rt = {chunk_title(c) for c in retrieved_chunks[:k]}
    hits = 0
    for st in supporting_titles:
        stl = st.strip().lower()
        if any(stl in t or t in stl for t in rt):
            hits += 1
    return hits / len(supporting_titles)
