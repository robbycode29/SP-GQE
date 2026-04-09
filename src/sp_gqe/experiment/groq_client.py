"""Groq OpenAI-compatible API with free-tier quota enforcement (RPM / TPM / RPD / TPD)."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sp_gqe.settings  # noqa: F401 — load `.env` before reading GROQ_API_KEY
from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# Defaults match Groq free plan for llama-3.1-8b-instant (see console limits table).
_DEFAULT_RPM = 30
_DEFAULT_TPM = 6000
_DEFAULT_RPD = 14400
_DEFAULT_TPD = 500_000

_groq_lock = threading.RLock()
_last_request_start: float = 0.0
_tpm_window: deque[tuple[float, int]] = deque()


class GroqQuotaExceeded(RuntimeError):
    """Raised when a daily or per-minute Groq limit would be exceeded."""


def _utc_date_str() -> str:
    return datetime.now(UTC).date().isoformat()


def _state_path() -> Path:
    from sp_gqe.settings import REPO_ROOT

    return REPO_ROOT / "data" / ".groq_quota_state.json"


def _load_daily_state() -> tuple[str, int, int]:
    """(date_utc, requests_recorded_here, tokens_recorded_here). Resets if date rolled over."""
    path = _state_path()
    today = _utc_date_str()
    if not path.exists():
        return today, 0, 0
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        d = str(raw.get("date_utc", ""))
        if d != today:
            return today, 0, 0
        return today, int(raw.get("requests", 0)), int(raw.get("tokens", 0))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return today, 0, 0


def _save_daily_state(date_str: str, requests: int, tokens: int) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"date_utc": date_str, "requests": requests, "tokens": tokens},
            indent=2,
        ),
        encoding="utf-8",
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    return int(raw)


def _initial_tokens_today() -> int:
    return _env_int("GROQ_TOKENS_USED_TODAY_INITIAL", 0)


def _initial_requests_today() -> int:
    return _env_int("GROQ_REQUESTS_USED_TODAY_INITIAL", 0)


def _max_rpd() -> int:
    return _env_int("GROQ_MAX_REQUESTS_PER_DAY", _DEFAULT_RPD)


def _max_tpd() -> int:
    return _env_int("GROQ_MAX_TOKENS_PER_DAY", _DEFAULT_TPD)


def _max_tpm() -> int:
    return _env_int("GROQ_MAX_TOKENS_PER_MINUTE", _DEFAULT_TPM)


def _tpd_safety_margin() -> int:
    """Stay this many tokens under TPD to avoid edge races."""
    return max(0, _env_int("GROQ_TPD_SAFETY_MARGIN", 2000))


def _min_interval_sec() -> float:
    explicit = os.environ.get("GROQ_MIN_INTERVAL_SEC", "").strip()
    if explicit != "":
        return max(0.0, float(explicit))
    rpm = float(os.environ.get("GROQ_MAX_REQUESTS_PER_MINUTE", str(_DEFAULT_RPM)))
    if rpm <= 0:
        return 0.0
    return 60.0 / rpm


def _prune_tpm_window_unlocked() -> None:
    cutoff = time.monotonic() - 60.0
    while _tpm_window and _tpm_window[0][0] < cutoff:
        _tpm_window.popleft()


def _tpm_sum_unlocked() -> int:
    _prune_tpm_window_unlocked()
    return sum(t for _, t in _tpm_window)


def _estimate_prompt_tokens(text: str) -> int:
    # Rough heuristic (~4 chars/token); good enough for throttling.
    return max(1, len(text) // 4)


def _throttle_rpm() -> None:
    interval = _min_interval_sec()
    if interval <= 0:
        return
    global _last_request_start
    with _groq_lock:
        now = time.monotonic()
        wait = interval - (now - _last_request_start)
        if wait > 0:
            time.sleep(wait)
        _last_request_start = time.monotonic()


def _check_daily_limits_pre_request(estimated_total_tokens: int) -> None:
    date_str, req_here, tok_here = _load_daily_state()
    total_req = _initial_requests_today() + req_here
    total_tok = _initial_tokens_today() + tok_here
    max_r = _max_rpd()
    max_t = _max_tpd() - _tpd_safety_margin()
    if total_req >= max_r:
        raise GroqQuotaExceeded(
            f"Groq RPD limit: {total_req}/{max_r} requests used today (initial + this run). "
            "Wait until UTC midnight or raise GROQ_MAX_REQUESTS_PER_DAY if your plan differs."
        )
    if total_tok + estimated_total_tokens > max_t:
        raise GroqQuotaExceeded(
            f"Groq TPD limit: ~{total_tok} tokens used today + ~{estimated_total_tokens} estimated "
            f"would exceed safe cap {max_t} (TPD {_max_tpd()} minus margin). "
            "Lower sample size or continue tomorrow."
        )


def _tpm_wait_until_room(estimated_tokens: int) -> None:
    max_tpm = _max_tpm()
    if max_tpm <= 0:
        return
    while True:
        with _groq_lock:
            used = _tpm_sum_unlocked()
        if used + estimated_tokens <= max_tpm:
            return
        time.sleep(0.25)


def _record_success(usage_tokens: int) -> None:
    with _groq_lock:
        date_str, req_here, tok_here = _load_daily_state()
        if date_str != _utc_date_str():
            req_here, tok_here = 0, 0
            date_str = _utc_date_str()
        req_here += 1
        tok_here += usage_tokens
        _save_daily_state(date_str, req_here, tok_here)
        _tpm_window.append((time.monotonic(), usage_tokens))


def groq_model() -> str:
    return os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL


def _client() -> OpenAI:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL)


def groq_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def groq_generate(
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    est = _estimate_prompt_tokens(prompt) + max_tokens
    with _groq_lock:
        _check_daily_limits_pre_request(est)
    _tpm_wait_until_room(est)
    _throttle_rpm()

    client = _client()
    r = client.chat.completions.create(
        model=groq_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = r.choices[0].message
    text = (msg.content or "").strip()

    usage = getattr(r, "usage", None)
    total_tok = 0
    if usage is not None:
        total_tok = int(getattr(usage, "total_tokens", None) or 0)
    if total_tok <= 0:
        total_tok = _estimate_prompt_tokens(prompt) + _estimate_prompt_tokens(text)

    _record_success(total_tok)
    return text


def answer_with_groq(question: str, contexts: list[str]) -> str:
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
    text = groq_generate(prompt, temperature=0.0)
    return text.split("\n")[0].strip()[:500]
