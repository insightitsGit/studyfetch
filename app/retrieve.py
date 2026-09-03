from __future__ import annotations

import json
import re
import time
from typing import Any

from app.db.store import Store
from app.embeddings import embed_query
from app.pipelines.prism import PrismShield, verify_signature, parameter_payload, load_or_create_key, public_key_hex
from app.vectorprism import CHANNELS, TABLES, fuse as vectorprism_fuse


def hybrid_merge(vec_hits: list[dict], fts_hits: list[dict], k: int) -> list[dict]:
    scored: dict[str, dict] = {}
    for rank, hit in enumerate(vec_hits):
        cid = hit.get("chunk_id") or hit.get("id")
        item = dict(hit)
        item["chunk_id"] = cid
        item["vec_rank"] = rank
        item["vec_distance"] = hit.get("distance")
        item["score"] = 1.0 / (1 + rank)
        scored[cid] = item
    for rank, hit in enumerate(fts_hits):
        cid = hit.get("chunk_id") or hit.get("id")
        if cid in scored:
            scored[cid]["score"] += 0.8 / (1 + rank)
            scored[cid]["fts_rank"] = rank
        else:
            item = dict(hit)
            item["chunk_id"] = cid
            item["fts_rank"] = rank
            item["score"] = 0.8 / (1 + rank)
            scored[cid] = item
    return sorted(scored.values(), key=lambda x: x["score"], reverse=True)[:k]


def retrieve(
    store: Store,
    query: str,
    pipeline_id: str,
    k: int = 6,
    intent: str | None = None,
    apply_shield: bool = True,
    document_id: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    qvec = embed_query(query)
    fts_hits = store.fts(query, pipeline_id, k=k, document_id=document_id)
    vectorprism = None

    if pipeline_id == "baseline":
        vec_hits = store.knn("vec_baseline", qvec, k=k, document_id=document_id)
        merged = hybrid_merge(vec_hits, fts_hits, k)
        index_name = "vec_baseline+fts5"
    elif pipeline_id == "relay":
        vec_hits = store.knn("vec_relay", qvec, k=k, document_id=document_id)
        merged = hybrid_merge(vec_hits, fts_hits, k)
        index_name = "vec_relay+fts5"
    elif pipeline_id == "prism":
        channel_hits = {
            ch: store.knn(TABLES[ch], qvec, k=k, document_id=document_id) for ch in CHANNELS
        }
        merged, vectorprism = vectorprism_fuse(channel_hits, fts_hits, intent, k)
        extra = _graph_expand(store, merged, document_id=document_id)
        merged = merge_graph_hits(merged, extra)[: k + 2]
        merged = boost_hits_with_signed_params(store, merged, query)
        if apply_shield:
            merged = PrismShield(store).filter_chunks(merged)
            merged = _annotate_verified_params(store, merged)
        index_name = "vectorprism:6ch+fts5+chorusgraph"
    else:
        raise ValueError(pipeline_id)

    if document_id:
        merged = [h for h in merged if h.get("document_id") == document_id]
    latency = (time.perf_counter() - t0) * 1000
    filenames = {
        r["id"]: r["filename"]
        for r in store.fetchall("SELECT id, filename FROM documents")
    }
    for item in merged:
        item["filename"] = filenames.get(item.get("document_id") or "", "")
        if isinstance(item.get("context_json"), str):
            try:
                item["context"] = json.loads(item["context_json"])
            except json.JSONDecodeError:
                item["context"] = {}
        if isinstance(item.get("asset_ids_json"), str):
            try:
                item["asset_ids"] = json.loads(item["asset_ids_json"])
            except json.JSONDecodeError:
                item["asset_ids"] = []
    _attach_page_labels(store, merged)
    payload = {
        "pipeline_id": pipeline_id,
        "index_name": index_name,
        "query": query,
        "intent": intent,
        "document_id": document_id,
        "latency_ms": round(latency, 2),
        "hits": merged,
        "design": design_card(pipeline_id, vectorprism),
    }
    if pipeline_id == "prism":
        payload["vectorprism"] = vectorprism
        payload["shield"] = _shield_summary(merged, apply_shield)
    else:
        payload["shield"] = {
            "applied": False,
            "role": "This stack does not run PrismShield. Metrics are unsigned prose.",
            "verified": [],
            "unsigned": [],
            "drifted": [],
        }
    return payload


def design_card(pipeline_id: str, vectorprism: dict | None = None) -> dict[str, Any]:
    """What this index is for — attached to every /api/query so Compare retrieval can show it."""
    cards = {
        "baseline": {
            "name": "Baseline · LangGraph control",
            "index": "vec_baseline + FTS5 (one semantic channel)",
            "chunk": "Recursive ~1100c window under a heading. Prefix is the inherited path only.",
            "useful": "RAG-ready outline + chunks with section path. No signed digits, no graph.",
            "trace": "document_id → filename/sha256; section_id; page_start/end on every hit.",
            "context": "Path prefix only. Does not paste sibling sections or the rest of the chapter.",
        },
        "prism": {
            "name": "Prism · VectorPrism 6ch + ChorusGraph",
            "index": "vectorprism:6ch+fts5+chorusgraph",
            "chunk": "Section window ~900c. Prefix = Document / Intent / Section path.",
            "useful": "Six subspaces + signed parameters + cross-doc edges a downstream agent can trust.",
            "trace": "pages + section_id + provenance_page on each signed metric + graph document ids.",
            "context": "Path + intent in the prefix. Figures stay as asset ids. Graph expand adds a related section, not the whole PDF.",
            "shield": "Query-time PrismShield: classifies numbers as verified / unsigned / drifted. Does not rewrite prose. Manifest is signed at ingest.",
        },
        "relay": {
            "name": "Relay · page-routed",
            "index": "vec_relay + FTS5",
            "chunk": "One leaf section (split only if long). Prefix = Document / Section / Pages.",
            "useful": "Page labels + asset_ids so a study product can cite the figure, not paste pixels.",
            "trace": "document_id, section_id, page range, page_label, asset_ids → /api/assets/{id}.",
            "context": "Whole leaf section, not a sliding window across headings. No sibling chapter text.",
        },
    }
    card = dict(cards.get(pipeline_id) or {"name": pipeline_id})
    if pipeline_id == "prism" and vectorprism:
        card["mix"] = vectorprism.get("weights")
        card["channels"] = vectorprism.get("channels")
    return card


def _shield_summary(hits: list[dict], applied: bool) -> dict[str, Any]:
    """Roll hit-level Shield audits into one query-time verdict for the UI."""
    verified: list[str] = []
    unsigned: list[str] = []
    drifted: list[str] = []
    for hit in hits:
        report = hit.get("shield") or {}
        verified.extend(report.get("verified_parameters") or [])
        unsigned.extend(report.get("unsigned") or [])
        for item in report.get("drifted") or []:
            drifted.append(item.get("raw") if isinstance(item, dict) else str(item))
    uniq = lambda xs: list(dict.fromkeys(x for x in xs if x))
    return {
        "applied": bool(applied),
        "role": "Query-time gate against the Ed25519 manifest. Flags same-unit inventions. Does not rewrite prose.",
        "verified": uniq(verified)[:12],
        "unsigned": uniq(unsigned)[:8],
        "drifted": uniq(drifted)[:8],
    }


def _attach_page_labels(store: Store, hits: list[dict]) -> None:
    if not hits:
        return
    docs = {h.get("document_id") for h in hits if h.get("document_id")}
    labels: dict[tuple[str, int], str] = {}
    for doc_id in docs:
        for row in store.fetchall(
            "SELECT page_number, label FROM pages WHERE document_id=?",
            (doc_id,),
        ):
            labels[(doc_id, int(row["page_number"]))] = row.get("label") or ""
    for h in hits:
        page = h.get("page_start")
        if page is None:
            continue
        h["page_label"] = labels.get((h.get("document_id"), int(page)))


_PARAM_WORD = re.compile(r"[a-z0-9]+")
_PARAM_STOP = {
    "what", "whatis", "with", "from", "that", "this", "should", "about",
    "document", "documents", "discuss", "where",
}


def param_name_overlap(query: str, param_name: str) -> int:
    """Shared content stems between a question and a bound parameter name."""
    return len(_stems(query) & _stems(param_name))


def _stems(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _PARAM_WORD.findall((text or "").lower()):
        if len(raw) < 4 or raw in _PARAM_STOP:
            continue
        word = raw
        if word.endswith("ing") and len(word) > 6:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 5:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        out.add(word)
    return out


def boost_hits_with_signed_params(store: Store, hits: list[dict], query: str) -> list[dict]:
    """Use the Ed25519 parameter table for ranking, not only Shield labels.

    A hit is boosted when a signed name on that document shares two or more
    content words with the question (commission+voltage vs maximum+operating).
    Baseline and Relay never see this table.
    """
    if not hits or not (query or "").strip():
        return hits
    docs = {h.get("document_id") for h in hits if h.get("document_id")}
    if not docs:
        return hits
    placeholders = ",".join("?" * len(docs))
    rows = store.fetchall(
        f"""
        SELECT document_id, parameter_name, raw_string_value, provenance_page
        FROM document_parameters
        WHERE document_id IN ({placeholders})
          AND COALESCE(manifest_signature, '') != ''
        """,
        tuple(docs),
    )
    by_doc: dict[str, list[dict]] = {}
    for row in rows:
        by_doc.setdefault(row["document_id"], []).append(row)
    if not by_doc:
        return hits

    scored = []
    for hit in hits:
        item = dict(hit)
        best = 0
        matched = None
        page = item.get("page_start")
        for param in by_doc.get(item.get("document_id") or "", []):
            overlap = param_name_overlap(query, param.get("parameter_name") or "")
            if overlap < 2:
                continue
            if page is not None and param.get("provenance_page") == page:
                overlap += 1
            if overlap > best:
                best = overlap
                matched = param
        if best >= 2 and matched is not None:
            item["score"] = float(item.get("score") or 0) + 0.35 * best
            item["param_boost"] = {
                "name": matched.get("parameter_name"),
                "value": matched.get("raw_string_value"),
                "overlap": best,
            }
            chans = list(item.get("channels") or [])
            if "signed_param" not in chans:
                chans.append("signed_param")
            item["channels"] = chans
        scored.append(item)
    scored.sort(key=lambda h: float(h.get("score") or 0), reverse=True)
    return scored


def merge_graph_hits(merged: list[dict], extra: list[dict]) -> list[dict]:
    """Fold ChorusGraph neighbors into the hit list.

    Several edges can resolve to the same related chunk. Keep `by_id` in
    sync so a second mention annotates the first instead of KeyError.
    """
    out = list(merged)
    by_id: dict[str, dict] = {}
    for hit in out:
        cid = hit.get("chunk_id") or hit.get("id")
        if cid:
            hit["chunk_id"] = cid
            by_id[cid] = hit
    for item in extra:
        cid = item.get("chunk_id") or item.get("id")
        if not cid:
            continue
        item = dict(item)
        item["chunk_id"] = cid
        item.setdefault("channels", [])
        if "chorusgraph" not in item["channels"]:
            item["channels"].append("chorusgraph")
        existing = by_id.get(cid)
        if existing:
            existing.setdefault("channels", [])
            if "chorusgraph" not in existing["channels"]:
                existing["channels"].append("chorusgraph")
            existing["graph_edge"] = item.get("graph_edge") or existing.get("graph_edge")
            if item.get("graph_weight") is not None:
                prev = existing.get("graph_weight")
                existing["graph_weight"] = (
                    max(float(prev), float(item["graph_weight"]))
                    if prev is not None
                    else item["graph_weight"]
                )
            continue
        out.append(item)
        by_id[cid] = item
    return out


def _graph_expand(store: Store, hits: list[dict], document_id: str | None = None) -> list[dict]:
    extra = []
    seen_extra: set[str] = set()
    for hit in hits[:3]:
        sid = hit.get("section_id")
        if not sid:
            continue
        edges = store.fetchall(
            """
            SELECT * FROM chorusgraph_edges
            WHERE (source_node=? OR target_node=?)
              AND relationship_type IN ('overlaps_with', 'same_entity')
            ORDER BY weight DESC LIMIT 3
            """,
            (f"node_{sid}", f"node_{sid}"),
        )
        for e in edges:
            other = e["target_node"] if e["source_node"] == f"node_{sid}" else e["source_node"]
            if not other.startswith("node_sec_") and not other.startswith("node_sec"):
                # section ids are sec_...
                other_id = other.removeprefix("node_")
            else:
                other_id = other.removeprefix("node_")
            chunk = store.fetchone(
                "SELECT * FROM chunks WHERE section_id=? AND pipeline_id='prism' ORDER BY chunk_index LIMIT 1",
                (other_id,),
            )
            if chunk and document_id and chunk["document_id"] != document_id:
                continue
            if not chunk:
                continue
            cid = chunk["id"]
            if cid in seen_extra:
                continue
            seen_extra.add(cid)
            hit.setdefault("channels", [])
            if "chorusgraph" not in hit["channels"]:
                hit["channels"].append("chorusgraph")
            hit["graph_edge"] = e["relationship_type"]
            hit["graph_weight"] = e["weight"]
            extra.append(
                {
                    **chunk,
                    "chunk_id": cid,
                    "score": 0.25 * float(e["weight"]),
                    "graph_edge": e["relationship_type"],
                    "graph_weight": e["weight"],
                }
            )
    return extra


def _annotate_verified_params(store: Store, hits: list[dict]) -> list[dict]:
    key = load_or_create_key()
    pub = public_key_hex(key)
    for hit in hits:
        params = store.fetchall(
            "SELECT * FROM document_parameters WHERE document_id=?",
            (hit.get("document_id"),),
        )
        text = hit.get("text") or hit.get("retrieval_text") or ""
        shield_verified = " ".join(hit.get("shield", {}).get("verified_parameters") or [])
        verified = []
        for p in params:
            raw = p["raw_string_value"] or ""
            if raw not in text and p["parameter_name"] not in text and raw not in shield_verified:
                continue
            ok = verify_signature(pub, parameter_payload(p), p["manifest_signature"] or "")
            if ok:
                verified.append(
                    {
                        "name": p["parameter_name"],
                        "raw": p["raw_string_value"],
                        "numeric": p["numeric_value"],
                        "page": p["provenance_page"],
                    }
                )
        hit["verified_parameters"] = verified
    return hits
