from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

# Published list prices used only for *estimated* USD. Local FastEmbed/Tesseract are $0.
# gpt-4o-mini and gpt-4o vision figures are what we *would* pay if those paths ran.
USD_LLM_IN_PER_M = 0.15
USD_LLM_OUT_PER_M = 0.60
USD_VISION_PER_PAGE = 0.01
USD_EMBED_API_PER_M = 0.02

_local = threading.local()


def _bucket() -> dict[str, int]:
    if not hasattr(_local, "counts"):
        _local.counts = defaultdict(int)
    return _local.counts


def reset() -> None:
    _local.counts = defaultdict(int)


def add(kind: str, n: int = 1) -> None:
    if n:
        _bucket()[kind] += int(n)


def snapshot(page_count: int = 0) -> dict[str, Any]:
    calls = dict(_bucket())
    llm_in = calls.get("llm_tokens_in", 0)
    llm_out = calls.get("llm_tokens_out", 0)
    actual = (
        (llm_in / 1_000_000) * USD_LLM_IN_PER_M
        + (llm_out / 1_000_000) * USD_LLM_OUT_PER_M
        + calls.get("vision_pages", 0) * USD_VISION_PER_PAGE
    )
    naive_vision = page_count * USD_VISION_PER_PAGE
    return {
        "embed_docs": calls.get("embed_docs", 0),
        "embed_queries": calls.get("embed_queries", 0),
        "embed_tokens": calls.get("embed_tokens", 0),
        "ocr_pages": calls.get("ocr_pages", 0),
        "ocr_failed": calls.get("ocr_failed", 0),
        "llm_calls": calls.get("llm_calls", 0),
        "llm_tokens_in": llm_in,
        "llm_tokens_out": llm_out,
        "vision_pages": calls.get("vision_pages", 0),
        "ed25519_signs": calls.get("ed25519_signs", 0),
        "estimated_usd_actual": round(actual, 6),
        "estimated_usd_if_every_page_vision": round(naive_vision, 4),
        "usd_saved_vs_per_page_vision": round(max(0.0, naive_vision - actual), 4),
        "notes": (
            "Embeddings are local FastEmbed (bge-small) — $0. "
            "OCR is local Tesseract — $0. "
            "LLM/vision stay at 0 unless OPENAI_API_KEY is set. "
            "‘If every page vision’ is the cost we refused by routing."
        ),
    }
