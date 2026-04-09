"""Lightweight co-occurrence graph (paragraph-level entities)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def norm_entity(s: str) -> str:
    return " ".join(s.lower().split())


class CooccurrenceKG:
    """Undirected multi-graph represented as adjacency sets."""

    def __init__(self) -> None:
        self.adj: dict[str, set[str]] = defaultdict(set)

    def add_clique(self, entities: Iterable[str]) -> None:
        ents = [norm_entity(e) for e in entities if e and len(norm_entity(e)) > 1]
        for i, a in enumerate(ents):
            for b in ents[i + 1 :]:
                if a != b:
                    self.adj[a].add(b)
                    self.adj[b].add(a)

    def n_hop_neighbors(self, seeds: set[str], n: int) -> set[str]:
        if n <= 0:
            return set(seeds)
        frontier = {norm_entity(s) for s in seeds if s}
        seen = set(frontier)
        for _ in range(n):
            nxt: set[str] = set()
            for u in frontier:
                for v in self.adj.get(u, ()):
                    if v not in seen:
                        nxt.add(v)
                        seen.add(v)
            frontier = nxt
            if not frontier:
                break
        return seen

    def one_hop(self, nodes: set[str]) -> set[str]:
        out: set[str] = set()
        for u in nodes:
            out |= self.adj.get(u, set())
        return out
