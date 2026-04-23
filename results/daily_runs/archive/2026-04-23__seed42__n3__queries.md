# SPARQL query samples — 2026-04-23, seed 42 (n=3)

Qualitative review of the two-branch SP-GQE(n=2, τ=0.5) pipeline.
For each question we show:

- the **Branch 1** SPARQL query (structural n-hop traversal from seed entities) and the entities it returns;
- the **Branch 2** SPARQL query (keyword-driven lookup over `rdfs:label`) and the entities it returns;
- the **union** of branches, the per-candidate cosine similarity to the reunion `{question} ∪ noun_chunks`, and the set **kept at τ = 0.5**;
- the **gold supporting entities** extracted from the HotpotQA supporting facts and the per-branch precision/recall (graph-query validity ablation).

## 1. [comparison] What type of community does Bob Hope Airport and Boeing Field have in common?

- **qid:** `5a84f06f5542994c784dda92`
- **gold answer:** public
- **SP-GQE F1 / V-RAG F1:** 0.000 / 0.000
- **seed entities (spaCy NER, normalised):** ['bob hope airport', 'boeing field']
- **noun-chunk probes:** ['What type', 'community', 'Bob Hope Airport', 'Boeing Field']

### Branch 1 — structural n-hop (SPARQL)

```sparql
PREFIX spg: <http://spgqe.local/>
SELECT DISTINCT ?t WHERE {
  VALUES ?s { <http://spgqe.local/e#bob_hope_airport> <http://spgqe.local/e#boeing_field> }
  { ?s spg:coOccurs ?t } UNION
  { ?s spg:coOccurs/spg:coOccurs ?t }
}
```

- **returned (n=48):** ['17', '1936', '3', 'air force plant', 'amtrak', 'bfi', 'bob hope airport', 'boeing field', 'boeing plant 2', 'bur', 'burbank', 'burbank airport', 'burbank-bob hope airport', 'california', 'chu-lin', 'daily', 'faa', 'five miles', 'greenville', 'hollywood burbank airport', 'iata', 'jetblue airways', 'jetblue flight 292', 'john f. kennedy', 'kbfi']...

### Branch 2 — keyword / semantic (SPARQL)

```sparql
PREFIX spg: <http://spgqe.local/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?e ?label WHERE {
  ?e a spg:Entity ; rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "type") || CONTAINS(LCASE(STR(?label)), "community") || CONTAINS(LCASE(STR(?label)), "bob") || CONTAINS(LCASE(STR(?label)), "hope") || CONTAINS(LCASE(STR(?label)), "airport") || CONTAINS(LCASE(STR(?label)), "boeing") || CONTAINS(LCASE(STR(?label)), "field"))
}
```

- **returned (n=20):** ['bob hope', 'bob hope airport', 'bob hope patriotic hall', 'bob hope school', "bob hope's", 'boeing', 'boeing 737s', 'boeing field', 'boeing plant 2', 'burbank airport', 'burbank-bob hope airport', 'hollywood burbank airport', 'hope & company', 'judith richards hope', 'king county international airport', 'los angeles international airport', 'the boeing b-17 flying fortresses', 'the boeing corporation', 'tony hope', 'william e. boeing']

### Fusion and pruning

- **union |A ∪ B| = 60**
- **kept after τ = 0.5 (n = 16):** ['bob hope', 'bob hope airport', 'bob hope patriotic hall', 'bob hope school', "bob hope's", 'boeing', 'boeing 737s', 'boeing field', 'boeing plant 2', 'burbank airport', 'burbank-bob hope airport', 'hollywood burbank airport', 'king county international airport', 'the boeing b-17 flying fortresses', 'the boeing corporation', 'william e. boeing']
- **top-10 candidate similarities (max cosine vs. reunion `{question} ∪ probes`):**

  - `burbank-bob hope airport` — 0.845
  - `boeing` — 0.788
  - `the boeing corporation` — 0.731
  - `bob hope` — 0.724
  - `bob hope's` — 0.705
  - `bob hope school` — 0.666
  - `boeing plant 2` — 0.640
  - `william e. boeing` — 0.621
  - `bob hope patriotic hall` — 0.578
  - `boeing 737s` — 0.575

### Ground truth & validity

- **supporting titles:** ['Bob Hope Airport', 'Boeing Field']
- **supporting entities (spaCy NER of gold paragraphs):** ['3', 'bfi', 'bob hope airport', 'boeing', 'boeing field', 'bur', 'burbank', 'california', 'daily', 'faa', 'five miles', 'glendale', 'greater los angeles', 'griffith park', 'hollywood', 'hollywood burbank airport', 'iata', 'jetblue airways', 'kbfi', 'kcia', 'king county', 'king county international airport', 'los angeles', 'los angeles county', 'los angeles international airport']...

| Stage | Precision | Recall |
|-------|-----------|--------|
| Branch 1 (n-hop) | 0.479 | 0.719 |
| Branch 2 (keyword) | 0.350 | 0.219 |
| Union | 0.433 | 0.812 |
| Kept after τ | 0.375 | 0.188 |

**Manual review (tick as appropriate):**

- [ ] Branch 1 SPARQL is well-formed and returns ≥ 1 supporting entity
- [ ] Branch 2 SPARQL is well-formed and returns ≥ 1 supporting entity
- [ ] τ pruning removed clearly off-topic candidates
- [ ] τ pruning did not drop a gold supporting entity that was present in the union
- [ ] Any supporting entity was missing from both branches (graph-construction gap)

---

## 2. [comparison] Are both American Foxhound and Löwchen types of Foxhounds?

- **qid:** `5ae54b3d5542990ba0bbb260`
- **gold answer:** no
- **SP-GQE F1 / V-RAG F1:** 0.000 / 0.000
- **seed entities (spaCy NER, normalised):** ['american foxhound', 'foxhounds', 'löwchen']
- **noun-chunk probes:** ['American Foxhound and Löwchen types', 'Foxhounds']

### Branch 1 — structural n-hop (SPARQL)

```sparql
PREFIX spg: <http://spgqe.local/>
SELECT DISTINCT ?t WHERE {
  VALUES ?s { <http://spgqe.local/e#l_wchen> }
  { ?s spg:coOccurs ?t } UNION
  { ?s spg:coOccurs/spg:coOccurs ?t }
}
```

- **returned (n=12):** ['1945', '2012', 'american foxhound', 'foxhounds', 'french', 'george washington', 'german', 'löwchen', 'the american kennel club', 'the marquis de lafayette', 'the treeing walker coonhound', 'the united kennel club']

### Branch 2 — keyword / semantic (SPARQL)

```sparql
PREFIX spg: <http://spgqe.local/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?e ?label WHERE {
  ?e a spg:Entity ; rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "american") || CONTAINS(LCASE(STR(?label)), "foxhound") || CONTAINS(LCASE(STR(?label)), "wchen") || CONTAINS(LCASE(STR(?label)), "types") || CONTAINS(LCASE(STR(?label)), "foxhounds"))
}
```

- **returned (n=11):** ['american', 'international foxhound association', 'löwchen', 'tan foxhounds', 'tan virginia foxhound', 'the american foxhound', 'the american kennel club', 'the english foxhound', 'the international foxhound association', 'the masters of foxhounds association', 'the trigg foxhound']

### Fusion and pruning

- **union |A ∪ B| = 21**
- **kept after τ = 0.5 (n = 11):** ['american foxhound', 'foxhounds', 'international foxhound association', 'löwchen', 'tan foxhounds', 'tan virginia foxhound', 'the american foxhound', 'the english foxhound', 'the international foxhound association', 'the masters of foxhounds association', 'the trigg foxhound']
- **top-10 candidate similarities (max cosine vs. reunion `{question} ∪ probes`):**

  - `tan foxhounds` — 0.825
  - `the american foxhound` — 0.799
  - `the english foxhound` — 0.788
  - `the masters of foxhounds association` — 0.765
  - `the international foxhound association` — 0.711
  - `the trigg foxhound` — 0.670
  - `international foxhound association` — 0.659
  - `tan virginia foxhound` — 0.589
  - `the treeing walker coonhound` — 0.378
  - `the american kennel club` — 0.373

### Ground truth & validity

- **supporting titles:** ['American Foxhound', 'Löwchen']
- **supporting entities (spaCy NER of gold paragraphs):** ['american foxhound', 'french', 'german', 'löwchen', 'the american foxhound', 'the american kennel club', 'the english foxhound']

| Stage | Precision | Recall |
|-------|-----------|--------|
| Branch 1 (n-hop) | 0.417 | 0.714 |
| Branch 2 (keyword) | 0.364 | 0.571 |
| Union | 0.333 | 1.000 |
| Kept after τ | 0.364 | 0.571 |

**Manual review (tick as appropriate):**

- [ ] Branch 1 SPARQL is well-formed and returns ≥ 1 supporting entity
- [ ] Branch 2 SPARQL is well-formed and returns ≥ 1 supporting entity
- [ ] τ pruning removed clearly off-topic candidates
- [ ] τ pruning did not drop a gold supporting entity that was present in the union
- [ ] Any supporting entity was missing from both branches (graph-construction gap)

---

## 3. [bridge] In which county does the  8th Military Police Brigade  of the United States Army's Barracks located?

- **qid:** `5adbf44355429944faac23ce`
- **gold answer:** Honolulu
- **SP-GQE F1 / V-RAG F1:** 0.000 / 0.000
- **seed entities (spaCy NER, normalised):** ['8th military police brigade', 'barracks', "the united states army's"]
- **noun-chunk probes:** ['which', 'county', 'the  8th Military Police Brigade', "the United States Army's Barracks"]

### Branch 1 — structural n-hop (SPARQL)

```sparql
PREFIX spg: <http://spgqe.local/>
SELECT DISTINCT ?t WHERE {
  VALUES ?s { <http://spgqe.local/e#8th_military_police_brigade> }
  { ?s spg:coOccurs ?t } UNION
  { ?s spg:coOccurs/spg:coOccurs ?t }
}
```

- **returned (n=5):** ['8th military police brigade', 'barracks', 'the pacific ocean', "the united states army's", 'united states']

### Branch 2 — keyword / semantic (SPARQL)

```sparql
PREFIX spg: <http://spgqe.local/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?e ?label WHERE {
  ?e a spg:Entity ; rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "county") || CONTAINS(LCASE(STR(?label)), "military") || CONTAINS(LCASE(STR(?label)), "police") || CONTAINS(LCASE(STR(?label)), "brigade") || CONTAINS(LCASE(STR(?label)), "united") || CONTAINS(LCASE(STR(?label)), "states") || CONTAINS(LCASE(STR(?label)), "army") || CONTAINS(LCASE(STR(?label)), "barracks"))
}
```

- **returned (n=23):** ['16th military police brigade', '18th military police brigade', '49th military police brigade', '8th military police brigade', 'army', 'army national guard', 'military police brigade', 'schofield barracks', 'sustainment brigade', 'the 16th military police brigade', 'the 49th military police brigade', 'the 709th military police battalion', 'the 720th military police battalion', 'the 759th military police battalion', 'the 89th military police brigade', 'the 95th military police battalion', 'the city and county', 'the united states army', "u.s. army's", 'united states', 'united states army', 'united states army europe', 'united states army military police']

### Fusion and pruning

- **union |A ∪ B| = 26**
- **kept after τ = 0.5 (n = 23):** ['16th military police brigade', '18th military police brigade', '49th military police brigade', '8th military police brigade', 'army', 'army national guard', 'barracks', 'military police brigade', 'schofield barracks', 'sustainment brigade', 'the 16th military police brigade', 'the 49th military police brigade', 'the 709th military police battalion', 'the 720th military police battalion', 'the 759th military police battalion', 'the 89th military police brigade', 'the 95th military police battalion', 'the city and county', 'the united states army', "the united states army's", "u.s. army's", 'united states army', 'united states army military police']
- **top-10 candidate similarities (max cosine vs. reunion `{question} ∪ probes`):**

  - `the 89th military police brigade` — 0.873
  - `military police brigade` — 0.865
  - `the city and county` — 0.818
  - `the 16th military police brigade` — 0.787
  - `the 49th military police brigade` — 0.779
  - `16th military police brigade` — 0.771
  - `the 720th military police battalion` — 0.770
  - `49th military police brigade` — 0.756
  - `the 709th military police battalion` — 0.729
  - `18th military police brigade` — 0.727

### Ground truth & validity

- **supporting titles:** ['8th Military Police Brigade (United States)', 'Schofield Barracks']
- **supporting entities (spaCy NER of gold paragraphs):** ['1872', '8th', '8th military police brigade', 'american', 'august 1888 to september 1895', 'cdp', 'commanding general', 'hawaii', 'hawaiʻ i.', 'honolulu', 'john mcallister schofield', 'lake wilson', 'military police brigade', 'pearl harbor', 'schofield barracks', 'the city and county', 'the pacific ocean', 'the united states army', 'the wahiawa district', 'united states', 'united states army', 'wahiawā', 'wahiawā reservoir']

| Stage | Precision | Recall |
|-------|-----------|--------|
| Branch 1 (n-hop) | 0.600 | 0.130 |
| Branch 2 (keyword) | 0.304 | 0.304 |
| Union | 0.308 | 0.348 |
| Kept after τ | 0.261 | 0.261 |

**Manual review (tick as appropriate):**

- [ ] Branch 1 SPARQL is well-formed and returns ≥ 1 supporting entity
- [ ] Branch 2 SPARQL is well-formed and returns ≥ 1 supporting entity
- [ ] τ pruning removed clearly off-topic candidates
- [ ] τ pruning did not drop a gold supporting entity that was present in the union
- [ ] Any supporting entity was missing from both branches (graph-construction gap)

---
