"""Neo4j storage for per-question co-occurrence subgraph (dissertation plan: Bolt + Cypher)."""

from __future__ import annotations

from neo4j import Driver, GraphDatabase

from sp_gqe.experiment.kg import norm_entity as norm_e
from sp_gqe.experiment.nlp_utils import extract_entities


def connect_neo4j(uri: str, user: str, password: str) -> Driver:
    drv = GraphDatabase.driver(uri, auth=(user, password))
    drv.verify_connectivity()
    return drv


class Neo4jQuestionGraph:
    """Load entities/edges for one HotpotQA example under a stable qid."""

    def __init__(self, driver: Driver, qid: str) -> None:
        self._driver = driver
        self.qid = qid

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (n:Entity {qid: $qid}) DETACH DELETE n",
                qid=self.qid,
            )

    def load_from_example(self, nlp, example: dict) -> None:
        self.clear()
        pairs: list[list[str]] = []
        for para in example["context"]:
            title, sents = para[0], para[1]
            for sent in sents:
                ents = extract_entities(nlp, sent)
                ents = [norm_e(e) for e in ents if e.strip()]
                if len(ents) >= 2:
                    for i, a in enumerate(ents):
                        for b in ents[i + 1 :]:
                            if a != b:
                                pairs.append([a, b])
                elif len(ents) == 1:
                    t_ents = extract_entities(nlp, title)
                    for te in t_ents:
                        te = norm_e(te)
                        if te and te != ents[0]:
                            pairs.append([ents[0], te])
        if not pairs:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $pairs AS p
                WITH p[0] AS na, p[1] AS nb, $qid AS qid
                MERGE (a:Entity {qid: qid, name: na})
                MERGE (b:Entity {qid: qid, name: nb})
                MERGE (a)-[:CO_OCCURS]->(b)
                MERGE (b)-[:CO_OCCURS]->(a)
                """,
                qid=self.qid,
                pairs=pairs,
            )

    def n_hop_neighbors(self, seeds: set[str], n: int) -> set[str]:
        seeds_l = [norm_e(s) for s in seeds if s]
        if not seeds_l:
            return set()
        if n <= 0:
            return set(seeds_l)
        n = max(1, min(n, 5))
        cypher = f"""
        MATCH (s:Entity {{qid: $qid}})
        WHERE s.name IN $seeds
        MATCH p = (s)-[*1..{n}]-(t:Entity {{qid: $qid}})
        RETURN DISTINCT t.name AS name
        """
        with self._driver.session() as session:
            result = session.run(cypher, qid=self.qid, seeds=seeds_l)
            out = {r["name"] for r in result if r["name"]}
        return out | set(seeds_l)

    def one_hop(self, nodes: set[str]) -> set[str]:
        nodes_l = [norm_e(n) for n in nodes if n]
        if not nodes_l:
            return set()
        cypher = """
        MATCH (s:Entity {qid: $qid})-[:CO_OCCURS]-(t:Entity {qid: $qid})
        WHERE s.name IN $nodes
        RETURN DISTINCT t.name AS name
        """
        with self._driver.session() as session:
            result = session.run(cypher, qid=self.qid, nodes=nodes_l)
            return {r["name"] for r in result if r["name"]}
