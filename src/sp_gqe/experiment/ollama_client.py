"""Ollama HTTP API: embeddings (nomic-embed-text) and generation (Mistral)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import requests

DEFAULT_HOST = "http://127.0.0.1:11434"


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


def _embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED", "nomic-embed-text")


def _llm_model() -> str:
    return os.environ.get("OLLAMA_LLM", "mistral")


class OllamaEmbedder:
    """Drop-in replacement for sentence-transformers Embedder (encode + .dim)."""

    def __init__(self) -> None:
        self.host = _host()
        self.model = _embed_model()
        self.dim = self._probe_dim()

    def _probe_dim(self) -> int:
        v = self._embed_one("dimension probe")
        return len(v)

    def _embed_one(self, text: str) -> list[float]:
        url = f"{self.host}/api/embeddings"
        r = requests.post(
            url,
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError(f"Ollama embeddings missing 'embedding': {data}")
        return emb

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for t in texts:
            emb = self._embed_one(t[:32000])
            rows.append(emb)
        m = np.array(rows, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
        m = m / norms
        return m.astype(np.float32)


def ollama_generate(
    prompt: str,
    *,
    temperature: float = 0.0,
    num_ctx: int = 8192,
) -> str:
    url = f"{_host()}/api/generate"
    r = requests.post(
        url,
        json={
            "model": _llm_model(),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def answer_with_mistral(question: str, contexts: list[str]) -> str:
    if not contexts:
        return ""
    blocks = []
    for i, c in enumerate(contexts[:12]):
        blocks.append(f"[{i + 1}] {c[:2500]}")
    ctx = "\n\n".join(blocks)
    prompt = f"""You are a careful QA assistant. Answer using only the numbered passages below.
Reply with a short answer: a name, date, place, yes/no, or a brief phrase. No explanation.

Passages:
{ctx}

Question: {question}

Answer:"""
    text = ollama_generate(prompt, temperature=0.0)
    return text.split("\n")[0].strip()[:500]


def check_ollama() -> None:
    r = requests.get(f"{_host()}/api/tags", timeout=10)
    r.raise_for_status()


def ollama_available() -> bool:
    try:
        check_ollama()
        return True
    except OSError:
        return False
    except requests.RequestException:
        return False
