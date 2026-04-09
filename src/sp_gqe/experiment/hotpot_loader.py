"""Download and sample HotpotQA distractor dev set."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import requests

DEFAULT_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)


def download_hotpot_dev(
    dest: Path,
    url: str = DEFAULT_URL,
    timeout: int = 120,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_hotpot(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sample_questions(
    items: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    bridge = [x for x in items if x.get("type") == "bridge"]
    comp = [x for x in items if x.get("type") == "comparison"]
    rng.shuffle(bridge)
    rng.shuffle(comp)
    half = n // 2
    out = bridge[:half] + comp[: n - half]
    rng.shuffle(out)
    return out


def iter_chunks(
    example: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return (chunk_id, text) for each sentence in the question context."""
    chunks: list[tuple[str, str]] = []
    for pi, para in enumerate(example["context"]):
        title, sents = para[0], para[1]
        for si, sent in enumerate(sents):
            cid = f"p{pi}_s{si}"
            chunks.append((cid, f"{title}. {sent}".strip()))
    return chunks
