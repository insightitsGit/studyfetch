"""VectorPrism — six-channel subspace retrieval for the Prism stack.

Each chunk is embedded six times into separate sqlite-vec tables. A query
hits every channel, then intent-weighted rank fusion (plus FTS) produces
the hit list. This is the multi-channel tensor the design calls for;
sqlite-vec has no packed multi-channel type, so each subspace is a vec0 table.
"""

from __future__ import annotations

from typing import Any

from app.extract.pdf_common import extract_parameters

CHANNELS = (
    "semantic",
    "structural",
    "title",
    "entity",
    "numeric",
    "caption",
)

TABLES = {ch: f"vec_prism_{ch}" for ch in CHANNELS}

# Weights sum to 1.0. Intent moves mass onto the subspace that should win.
_WEIGHTS: dict[str, dict[str, float]] = {
    "parameter": {
        "semantic": 0.16,
        "structural": 0.18,
        "title": 0.10,
        "entity": 0.08,
        "numeric": 0.38,
        "caption": 0.10,
    },
    "financial": {
        "semantic": 0.16,
        "structural": 0.16,
        "title": 0.10,
        "entity": 0.10,
        "numeric": 0.38,
        "caption": 0.10,
    },
    "technical": {
        "semantic": 0.18,
        "structural": 0.18,
        "title": 0.12,
        "entity": 0.10,
        "numeric": 0.28,
        "caption": 0.14,
    },
    "academic": {
        "semantic": 0.28,
        "structural": 0.22,
        "title": 0.18,
        "entity": 0.16,
        "numeric": 0.06,
        "caption": 0.10,
    },
    "outline": {
        "semantic": 0.18,
        "structural": 0.28,
        "title": 0.28,
        "entity": 0.12,
        "numeric": 0.04,
        "caption": 0.10,
    },
    "default": {
        "semantic": 0.32,
        "structural": 0.16,
        "title": 0.14,
        "entity": 0.14,
        "numeric": 0.12,
        "caption": 0.12,
    },
}

FTS_WEIGHT = 0.28


def table_for(channel: str) -> str:
    if channel not in TABLES:
        raise ValueError(channel)
    return TABLES[channel]


def weights_for(intent: str | None) -> dict[str, float]:
    key = (intent or "").lower()
    return dict(_WEIGHTS.get(key) or _WEIGHTS["default"])


def channel_texts(
    *,
    title: str,
    intent: str,
    section_path: list[str],
    section_title: str,
    body: str,
    retrieval_text: str,
    entities: list[str],
    captions: list[str],
    page_start: int,
) -> dict[str, str]:
    path = " > ".join(section_path) if section_path else section_title or title
    leaf = section_path[-1] if section_path else section_title or title
    params = extract_parameters(body or section_title, page_start, None)
    numeric = "; ".join(
        f"{p['parameter_name']}={p['raw_string_value']}" for p in params
    ) or f"{title} {leaf}"
    entity_blob = ", ".join(entities) or leaf
    caption_blob = " | ".join(c for c in captions if c) or f"{title} p.{page_start}"
    return {
        "semantic": retrieval_text or body or section_title,
        "structural": f"{leaf} {path}",
        "title": f"{title} / {section_title or leaf}",
        "entity": entity_blob,
        "numeric": numeric,
        "caption": caption_blob,
    }


def fuse(
    channel_hits: dict[str, list[dict]],
    fts_hits: list[dict],
    intent: str | None,
    k: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Intent-weighted reciprocal-rank fusion across the six vec channels + FTS."""
    weights = weights_for(intent)
    scored: dict[str, dict] = {}
    for channel in CHANNELS:
        w = weights.get(channel, 0.0)
        for rank, hit in enumerate(channel_hits.get(channel) or []):
            cid = hit.get("chunk_id") or hit.get("id")
            if not cid:
                continue
            if cid not in scored:
                scored[cid] = dict(hit)
                scored[cid]["chunk_id"] = cid
                scored[cid]["score"] = 0.0
                scored[cid]["channels"] = []
                scored[cid]["channel_ranks"] = {}
            scored[cid]["score"] += w / (1 + rank)
            if channel not in scored[cid]["channels"]:
                scored[cid]["channels"].append(channel)
            scored[cid]["channel_ranks"][channel] = rank
    for rank, hit in enumerate(fts_hits):
        cid = hit.get("chunk_id") or hit.get("id")
        if not cid:
            continue
        if cid not in scored:
            scored[cid] = dict(hit)
            scored[cid]["chunk_id"] = cid
            scored[cid]["score"] = 0.0
            scored[cid]["channels"] = []
            scored[cid]["channel_ranks"] = {}
        scored[cid]["score"] += FTS_WEIGHT / (1 + rank)
        if "fts" not in scored[cid]["channels"]:
            scored[cid]["channels"].append("fts")
        scored[cid]["channel_ranks"]["fts"] = rank
    merged = sorted(scored.values(), key=lambda x: x["score"], reverse=True)[:k]
    meta = {
        "name": "VectorPrism",
        "channels": list(CHANNELS),
        "tables": [TABLES[c] for c in CHANNELS],
        "weights": weights,
        "fts_weight": FTS_WEIGHT,
        "hits_per_channel": {c: len(channel_hits.get(c) or []) for c in CHANNELS},
    }
    return merged, meta
