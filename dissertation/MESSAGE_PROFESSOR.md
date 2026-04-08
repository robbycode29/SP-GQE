# Mesaj pentru profesor (Teams)

---

Bună ziua, domnule profesor.

Am lucrat la direcția de cercetare pe care ați sugerat-o
(GraphRAG, knowledge graphs).
Am citit literatura recentă (surveys 2024–2025, ICLR, ACL)
și am formulat o propunere concretă.

**Tema:**
Expansiune grafică cu filtrare semantică
pentru Retrieval-Augmented Generation.

**Problema:**
Când extinzi căutarea într-un knowledge graph
cu mai mult de 1-2 hop-uri,
numărul de entități crește exponențial
și aduce zgomot în retrieval.
Lucrările existente rezolvă asta fie cu modele antrenate
(SubgraphRAG, ICLR 2025),
fie cu re-ranking post-retrieval.

**Ce propun:**
O metodă în două faze — „recall and verify":

1. Extind entitățile din întrebare prin traversare de graf
   și generez interogări structurate (Cypher) pe baza sub-query-urilor.
2. Filtrez entitățile expandate pe baza divergenței semantice
   față de sub-query-uri și rezultatele structurate.
   Entitățile irelevante sunt eliminate înainte de retrieval.

Am și o variantă iterativă: filtrarea se aplică la fiecare hop,
per nod — expansiunea crește doar pe ramuri relevante
și se oprește singură pe cele irelevante.
Perechea (adâncime, prag) devine auto-reglatoare.

**Evaluare:**
Compar 6 pipeline-uri pe un subset HotpotQA (150 întrebări),
cu metrici standard (F1, EM, RAGAS).
Totul local, open-source, fără costuri: Ollama, Neo4j, FAISS.

Am pregătit un document scurt cu abstract, state of the art
și arhitectura propusă — vi-l pot trimite imediat.

Când ați avea disponibilitate pentru o discuție scurtă?

Mulțumesc,
Robert
