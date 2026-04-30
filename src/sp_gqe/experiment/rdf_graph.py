"""In-memory RDF graph per question with SPARQL 1.1 access.

Triples written per question:
    <e> a spg:Entity .
    <e> rdfs:label "normalized surface form" .
    <a> spg:coOccurs <b> .       # stored both directions for symmetry

Two SPARQL queries are exposed and used by the SP-GQE two-branch pipeline:
    * `n_hop_neighbors`  — structural traversal up to n hops via property paths.
    * `keyword_entities` — semantic keyword-driven lookup over rdfs:label.

Both methods also expose a `build_*_sparql` sibling that returns the literal
SPARQL text used for the call, so we can log the query for the thesis' graph-
query validity ablation.
"""

from __future__ import annotations

import re
from typing import Iterable

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

from sp_gqe.experiment.kg import norm_entity
from sp_gqe.experiment.nlp_utils import extract_entities

SPG = Namespace("http://spgqe.local/")
ENT_NS = "http://spgqe.local/e#"

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]+")

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "what", "which", "who", "whom", "where", "when",
        "how", "this", "that", "there", "these", "those", "from", "was", "were",
        "are", "did", "does", "has", "have", "had", "been", "also", "with",
        "into", "over", "about", "their", "them", "they", "some", "more", "most",
        "than", "then", "only", "once", "just", "very", "other", "such",
    }
)


def _ent_uri(normalised_name: str) -> URIRef:
    slug = _SLUG_RE.sub("_", normalised_name.strip()).strip("_") or "e"
    return URIRef(f"{ENT_NS}{slug}")


class RdfQuestionGraph:
    """Per-question labelled property graph stored as RDF, queried via SPARQL."""

    def __init__(self) -> None:
        self.g: Graph = Graph()
        self.g.bind("spg", SPG)
        self.g.bind("rdfs", RDFS)
        self._name_to_uri: dict[str, URIRef] = {}
        self._uri_to_name: dict[URIRef, str] = {}

    # ----- construction -------------------------------------------------

    def clear(self) -> None:
        self.g.remove((None, None, None))
        self._name_to_uri.clear()
        self._uri_to_name.clear()

    def _ensure(self, name: str) -> URIRef:
        n = norm_entity(name)
        if not n:
            return URIRef(f"{ENT_NS}_empty")
        if n in self._name_to_uri:
            return self._name_to_uri[n]
        u = _ent_uri(n)
        self._name_to_uri[n] = u
        self._uri_to_name[u] = n
        self.g.add((u, RDF.type, SPG.Entity))
        self.g.add((u, RDFS.label, Literal(n)))
        return u

    def add_cooccurs(self, a: str, b: str) -> None:
        ua, ub = self._ensure(a), self._ensure(b)
        if ua == ub:
            return
        self.g.add((ua, SPG.coOccurs, ub))
        self.g.add((ub, SPG.coOccurs, ua))

    def load_from_example(self, nlp, example: dict) -> None:
        self.clear()
        for para in example.get("context", []) or []:
            if not para or len(para) < 2:
                continue
            title, sents = para[0], para[1]
            for sent in sents:
                ents = [norm_entity(e) for e in extract_entities(nlp, sent)]
                ents = [e for e in ents if e]
                if len(ents) >= 2:
                    for i, a in enumerate(ents):
                        for b in ents[i + 1 :]:
                            self.add_cooccurs(a, b)
                elif len(ents) == 1:
                    t_ents = [norm_entity(e) for e in extract_entities(nlp, title) if e]
                    for te in t_ents:
                        if te and te != ents[0]:
                            self.add_cooccurs(ents[0], te)

    # ----- SPARQL helpers -----------------------------------------------

    def _seed_uris(self, seeds: Iterable[str]) -> list[URIRef]:
        out: list[URIRef] = []
        for s in seeds:
            n = norm_entity(s)
            if n in self._name_to_uri:
                out.append(self._name_to_uri[n])
        return out

    @staticmethod
    def _keyword_tokens(keywords: Iterable[str], min_len: int, limit: int) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for kw in keywords or []:
            if not kw:
                continue
            for t in _WORD_RE.findall(kw.lower()):
                if len(t) < min_len or t in _STOPWORDS or t in seen:
                    continue
                seen.add(t)
                out.append(t)
                if len(out) >= limit:
                    return out
        return out

    # ----- Branch 1: structural n-hop ----------------------------------

    def build_n_hop_sparql(self, seeds: Iterable[str], n: int) -> str:
        seeds_l = {norm_entity(s) for s in seeds if s}
        uris = self._seed_uris(seeds_l)
        n = max(1, min(int(n), 5))
        values_clause = " ".join(f"<{u}>" for u in uris) or "<urn:spgqe:none>"
        path_alts = " UNION\n  ".join(
            "{ ?s " + "/".join(["spg:coOccurs"] * k) + " ?t }"
            for k in range(1, n + 1)
        )
        return (
            "PREFIX spg: <http://spgqe.local/>\n"
            "SELECT DISTINCT ?t WHERE {\n"
            f"  VALUES ?s {{ {values_clause} }}\n"
            f"  {path_alts}\n"
            "}"
        )

    def n_hop_neighbors(self, seeds: Iterable[str], n: int) -> set[str]:
        seeds_l = {norm_entity(s) for s in seeds if s}
        if int(n) <= 0:
            return set(seeds_l)
        uris = self._seed_uris(seeds_l)
        if not uris:
            return set(seeds_l)
        q = self.build_n_hop_sparql(seeds_l, n)
        out: set[str] = set(seeds_l)
        try:
            for row in self.g.query(q):
                u = row[0]
                if isinstance(u, URIRef) and u in self._uri_to_name:
                    out.add(self._uri_to_name[u])
        except Exception:
            return set(seeds_l)
        return out

    # ----- Branch 2: keyword / semantic SPARQL --------------------------

    def build_keyword_sparql(
        self,
        keywords: Iterable[str],
        min_len: int = 3,
        limit: int = 20,
    ) -> str:
        toks = self._keyword_tokens(keywords, min_len, limit)
        if not toks:
            filt = "false"
        else:
            filt = " || ".join(
                f'CONTAINS(LCASE(STR(?label)), "{t}")' for t in toks
            )
        return (
            "PREFIX spg: <http://spgqe.local/>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "SELECT DISTINCT ?e ?label WHERE {\n"
            "  ?e a spg:Entity ; rdfs:label ?label .\n"
            f"  FILTER({filt})\n"
            "}"
        )

    def cooccurrence_edges_among(self, names: set[str]) -> list[tuple[str, str]]:
        """Undirected `spg:coOccurs` edges with both endpoints in *names* (deduped)."""
        norm_names = {norm_entity(n) for n in names if n}
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for s, _p, o in self.g.triples((None, SPG.coOccurs, None)):
            if not isinstance(s, URIRef) or not isinstance(o, URIRef):
                continue
            a = self._uri_to_name.get(s)
            b = self._uri_to_name.get(o)
            if not a or not b or a == b:
                continue
            if a not in norm_names or b not in norm_names:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return sorted(out)

    def entity_uri_map(self, names: Iterable[str]) -> dict[str, str]:
        """IRI string per normalised label (for GraphDB-style captions)."""
        out: dict[str, str] = {}
        for n in names:
            nn = norm_entity(n)
            if nn in self._name_to_uri:
                out[nn] = str(self._name_to_uri[nn])
        return out

    def keyword_entities(
        self,
        keywords: Iterable[str],
        min_len: int = 3,
        limit: int = 20,
    ) -> set[str]:
        toks = self._keyword_tokens(keywords, min_len, limit)
        if not toks:
            return set()
        q = self.build_keyword_sparql(keywords, min_len=min_len, limit=limit)
        out: set[str] = set()
        try:
            for row in self.g.query(q):
                u = row[0]
                if isinstance(u, URIRef) and u in self._uri_to_name:
                    out.add(self._uri_to_name[u])
        except Exception:
            return set()
        return out

    # ----- one-hop (used by SP-GQE-i iterative variant) -----------------

    def one_hop(self, nodes: Iterable[str]) -> set[str]:
        nodes_l = {norm_entity(n) for n in nodes if n}
        uris = self._seed_uris(nodes_l)
        if not uris:
            return set()
        values_clause = " ".join(f"<{u}>" for u in uris)
        q = (
            "PREFIX spg: <http://spgqe.local/>\n"
            "SELECT DISTINCT ?t WHERE {\n"
            f"  VALUES ?s {{ {values_clause} }}\n"
            "  ?s spg:coOccurs ?t .\n"
            "}"
        )
        out: set[str] = set()
        try:
            for row in self.g.query(q):
                u = row[0]
                if isinstance(u, URIRef) and u in self._uri_to_name:
                    out.add(self._uri_to_name[u])
        except Exception:
            return set()
        return out
