"""Shared spaCy helpers (avoid circular imports)."""

from __future__ import annotations

from spacy.language import Language


def extract_entities(nlp: Language, text: str) -> list[str]:
    doc = nlp(text[:10000])
    return [e.text for e in doc.ents if e.text.strip()]


def noun_chunks(nlp: Language, text: str) -> list[str]:
    doc = nlp(text[:10000])
    return [nc.text.strip() for nc in doc.noun_chunks if len(nc.text.strip()) > 2]
