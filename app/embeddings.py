from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from app.config import settings
from app.usage import add as usage_add

_lock = threading.Lock()
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=settings.embedding_model)
        return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _load_model()
    usage_add("embed_docs", len(texts))
    usage_add("embed_tokens", sum(max(1, len(t) // 4) for t in texts))
    vectors = list(model.embed(texts))
    out: list[list[float]] = []
    for vec in vectors:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape[0] != settings.embedding_dim:
            raise RuntimeError(
                f"embedding dim {arr.shape[0]} != configured {settings.embedding_dim}"
            )
        out.append(arr.tolist())
    return out


@lru_cache(maxsize=256)
def embed_query(text: str) -> list[float]:
    from app.usage import add as usage_add

    usage_add("embed_queries", 1)
    return embed_texts([text])[0]


def warmup() -> dict:
    vec = embed_texts(["document intelligence warmup"])[0]
    return {"model": settings.embedding_model, "dim": len(vec)}
