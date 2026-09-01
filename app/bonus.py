"""Cross-document bonus: related sections, shared concepts, unique content."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.db.store import Store


def analyze_collection(store: Store) -> dict[str, Any]:
    docs = store.fetchall("SELECT id, filename, title FROM documents")
    names = {d["id"]: d.get("title") or d["filename"] for d in docs}
    files = {d["id"]: d["filename"] for d in docs}
    nodes = {n["node_id"]: n for n in store.fetchall("SELECT * FROM chorusgraph_nodes")}
    sections = store.fetchall(
        "SELECT id, document_id, title, summary, page_start FROM sections WHERE pipeline_id='prism'"
    )
    sec_by_id = {s["id"]: s for s in sections}
    edges = store.fetchall(
        """
        SELECT * FROM chorusgraph_edges
        WHERE relationship_type IN ('overlaps_with', 'same_entity')
        """
    )

    related = []
    overlapped_sec: set[str] = set()
    pair_scores: dict[tuple[str, str], list[float]] = defaultdict(list)

    for e in edges:
        src_doc = e.get("document_id_source")
        tgt_doc = e.get("document_id_target")
        if not src_doc or not tgt_doc or src_doc == tgt_doc:
            continue
        src_node = nodes.get(e["source_node"]) or {}
        tgt_node = nodes.get(e["target_node"]) or {}
        src_sec = _section_from_node(e["source_node"], sec_by_id)
        tgt_sec = _section_from_node(e["target_node"], sec_by_id)
        if e["relationship_type"] == "overlaps_with":
            if src_sec:
                overlapped_sec.add(src_sec["id"])
            if tgt_sec:
                overlapped_sec.add(tgt_sec["id"])
        key = tuple(sorted((src_doc, tgt_doc)))
        pair_scores[key].append(float(e.get("weight") or 0))
        related.append(
            {
                "relationship": e["relationship_type"],
                "weight": e.get("weight"),
                "why": _why(e["relationship_type"], e.get("weight"), src_node, tgt_node),
                "source": _end(src_doc, names, files, src_node, src_sec),
                "target": _end(tgt_doc, names, files, tgt_node, tgt_sec),
            }
        )
    related.sort(key=lambda r: float(r.get("weight") or 0), reverse=True)

    unique = []
    for s in sections:
        if s["id"] in overlapped_sec:
            continue
        unique.append(
            {
                "document_id": s["document_id"],
                "filename": files.get(s["document_id"]),
                "title": s["title"],
                "page_start": s["page_start"],
                "preview": (s.get("summary") or "")[:160],
            }
        )

    pairs = []
    for (a, b), weights in pair_scores.items():
        pairs.append(
            {
                "document_a": {"id": a, "filename": files.get(a), "title": names.get(a)},
                "document_b": {"id": b, "filename": files.get(b), "title": names.get(b)},
                "edge_count": len(weights),
                "mean_weight": round(sum(weights) / len(weights), 3),
                "max_weight": round(max(weights), 3),
                "level": "section",
            }
        )
    pairs.sort(key=lambda p: p["max_weight"], reverse=True)

    return {
        "documents": len(docs),
        "related_sections": related[:40],
        "related_count": len(related),
        "document_pairs": pairs,
        "unique_sections": unique[:40],
        "unique_count": len(unique),
        "note": (
            "overlaps_with = section title+summary cosine ≥ 0.70 (different wording). "
            "same_entity = shared capitalized concept. "
            "unique = Prism sections with no cross-doc overlap edge."
        ),
    }


def _section_from_node(node_id: str, sec_by_id: dict) -> dict | None:
    sid = (node_id or "").removeprefix("node_")
    return sec_by_id.get(sid)


def _end(doc_id: str, names: dict, files: dict, node: dict, section: dict | None) -> dict:
    return {
        "document_id": doc_id,
        "filename": files.get(doc_id),
        "document": names.get(doc_id),
        "label": (section or {}).get("title") or node.get("label") or node.get("node_id"),
        "node_type": node.get("node_type"),
        "page_start": (section or {}).get("page_start"),
    }


def _why(rel: str, weight: Any, src: dict, tgt: dict) -> str:
    w = f"{float(weight or 0):.2f}"
    if rel == "same_entity":
        label = src.get("label") or tgt.get("label") or "concept"
        return f"Shared concept “{label}” (exact entity label, weight {w})."
    if rel == "overlaps_with":
        return (
            f"Section meaning overlaps (cosine {w} on title+summary), "
            "not keyword match."
        )
    return f"{rel} weight {w}"
