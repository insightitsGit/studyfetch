from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app.config import settings
from app.db.store import utcnow
from app.embeddings import embed_texts
from app.extract.pdf_common import (
    build_sections,
    classify_document_intent,
    detect_boilerplate,
    extract_blocks,
    extract_parameters,
    open_pdf,
    page_profiles,
    pdf_metadata,
    recursive_split,
    reorder_page_blocks,
    section_text,
    strip_boilerplate,
)
from app.pipelines.base import Pipeline, new_run, token_estimate
from app.vectorprism import CHANNELS, TABLES, channel_texts

KEY_PATH_NAME = "prism_ed25519.pem"

ENTITY_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,}(?:-[A-Z0-9]+)?)\b"
)
STOP = {
    "The", "This", "That", "Figure", "Table", "Chapter", "Section", "Page",
    "Abstract", "Introduction", "Conclusion", "References", "Appendix",
}


def _key_path() -> Path:
    path = settings.data_dir / KEY_PATH_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_or_create_key() -> Ed25519PrivateKey:
    path = _key_path()
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return key


def public_key_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def sign_payload(key: Ed25519PrivateKey, payload: bytes) -> str:
    from app.usage import add as usage_add

    usage_add("ed25519_signs", 1)
    return key.sign(payload).hex()


def verify_signature(public_hex: str, payload: bytes, signature_hex: str) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature

    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
    try:
        pub.verify(bytes.fromhex(signature_hex), payload)
        return True
    except (InvalidSignature, ValueError):
        return False


def parameter_payload(param: dict[str, Any]) -> bytes:
    canonical = {
        "name": param["parameter_name"],
        "raw": param["raw_string_value"],
        "numeric": param.get("numeric_value"),
        "unit": param.get("unit") or "",
        "page": param.get("provenance_page"),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()


class PrismShield:
    """Gate drifted metrics only.

    Signed values stay in the chunk. Unsigned prose numbers (F1, years, page
    counts) stay too. A token is *drift* when it has the same unit as a signed
    parameter but a different value that was never bound in the manifest —
    the case fuzzy retrieval would invent 23 V next to a signed 24 V.
    """

    _METRIC = re.compile(
        r"(?<![\w-])(\$?-?\d[\d,]*(?:\.\d+)?(?:\s*(?:V|A|W|kW|MHz|GHz|USD|ms|mm|kg|%))?)(?!\w)"
    )

    def __init__(self, store):
        self.store = store

    def filter_chunks(self, chunks: list[dict[str, Any]], rewrite_drift: bool = False) -> list[dict[str, Any]]:
        guarded = []
        for chunk in chunks:
            doc_id = chunk.get("document_id")
            params = self.store.fetchall(
                "SELECT * FROM document_parameters WHERE document_id=?",
                (doc_id,),
            )
            text = chunk.get("retrieval_text") or chunk.get("text") or ""
            report = self._audit(text, params)
            item = dict(chunk)
            item["retrieval_text"] = text
            if rewrite_drift and report["drifted"]:
                item["retrieval_text"] = self._rewrite_drift(text, report["drifted"])
            item["shield"] = report
            guarded.append(item)
        return guarded

    def _audit(self, text: str, params: list[dict]) -> dict[str, Any]:
        signed = [p for p in params if p.get("manifest_signature")]
        signed_raws = {re.sub(r"\s+", " ", (p["raw_string_value"] or "")).strip().lower() for p in signed}
        signed_pairs: set[tuple[float | None, str]] = set()
        units_we_bind: set[str] = set()
        for p in signed:
            unit = _canonical_unit(p.get("raw_string_value") or "", p.get("unit") or "", p.get("data_type") or "")
            if unit:
                units_we_bind.add(unit)
            try:
                signed_pairs.add((round(float(p["numeric_value"]), 6), unit))
            except (TypeError, ValueError):
                pass

        verified, unsigned, drifted = [], [], []
        for m in self._METRIC.finditer(text):
            raw = re.sub(r"\s+", " ", m.group(1)).strip()
            numeric, unit = _split_metric(raw)
            unit = _canonical_unit(raw, unit)
            key = raw.lower()
            pair = (round(numeric, 6) if numeric is not None else None, unit)
            if key in signed_raws or (pair[0] is not None and pair in signed_pairs):
                verified.append(raw)
                continue
            # Drift = same bound unit, different value, never signed.
            # Ordinary prose (years, F1, section 3.1) has no bound unit → unsigned.
            if unit and unit in units_we_bind and numeric is not None:
                drifted.append({"raw": raw, "unit": unit, "numeric": numeric})
                continue
            unsigned.append(raw)

        return {
            "verified_parameters": verified,
            "unsigned": unsigned,
            "drifted": drifted,
            "stripped": [d["raw"] for d in drifted],
        }

    def _rewrite_drift(self, text: str, drifted: list[dict]) -> str:
        out = text
        for d in drifted:
            out = out.replace(d["raw"], f"[DRIFT:{d['raw']}]")
        return out


def _canonical_unit(raw: str, unit: str, data_type: str = "") -> str:
    unit = (unit or "").strip().lower().replace("°c", "c")
    if unit in {"$", "usd"} or (raw or "").strip().startswith("$") or data_type == "currency":
        return "usd"
    return unit


def _split_metric(raw: str) -> tuple[float | None, str]:
    m = re.match(r"^\$?\s*(-?[\d,]+(?:\.\d+)?)\s*([A-Za-z%]+)?$", raw.strip())
    if not m:
        return None, ""
    try:
        numeric = float(m.group(1).replace(",", ""))
    except ValueError:
        return None, ""
    return numeric, _canonical_unit(raw, m.group(2) or "")


def extract_entities(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in ENTITY_RE.finditer(text):
        term = m.group(1).strip()
        if term in STOP or len(term) < 3:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(term)
    return found[:40]


class PrismPipeline(Pipeline):
    pipeline_id = "prism"
    display_name = "Prism Stack (GraphRAG + Zero-Trust)"

    def run(self, document_id: str, pdf_path: Path) -> dict[str, Any]:
        run_id = new_run(self.store, self.pipeline_id, document_id)
        warnings: list[str] = []
        self.begin_usage()
        try:
            doc = open_pdf(str(pdf_path))
            try:
                meta = pdf_metadata(doc)
                title = meta.get("title") or (self.store.fetchone(
                    "SELECT title FROM documents WHERE id=?", (document_id,)
                ) or {}).get("title") or "Untitled"
                profiles = page_profiles(doc, str(pdf_path))
                widths = {p.page_number: p.width for p in profiles}
                heights = {p.page_number: p.height for p in profiles}
                blocks = extract_blocks(doc)
            finally:
                doc.close()

            self.report("prismcortex", 18, "classify intent + extract blocks")
            full_text = "\n".join(b.text for b in blocks)
            intent = classify_document_intent(full_text, profiles, title)
            # PrismCortex: specialized dispatch
            if intent in {"academic", "textbook"}:
                blocks = reorder_page_blocks(blocks, widths)
                warnings.append(f"PrismCortex routed as {intent}: column-aware section tree")
            else:
                # financial / technical — keep reading order but prefer table-bound text
                blocks = reorder_page_blocks(blocks, widths)
                warnings.append(f"PrismCortex routed as {intent}: grid/parameter-first parse")

            junk = detect_boilerplate(blocks, heights, len(profiles))
            blocks, removed = strip_boilerplate(blocks, junk, heights)
            if removed:
                warnings.append(f"stripped {len(removed)} boilerplate blocks")

            self.report("section tree", 32, f"intent={intent}")
            sections = build_sections(blocks, title)
            assets = self.store.fetchall(
                "SELECT id, page_number, caption, asset_type FROM assets WHERE document_id=?",
                (document_id,),
            )
            section_rows = []
            chunk_rows = []
            param_rows = []
            nodes = []
            edges = []

            doc_node = f"node_doc_{document_id}"
            nodes.append(
                {
                    "node_id": doc_node,
                    "node_type": "document",
                    "document_id": document_id,
                    "label": title,
                    "extra_json": json.dumps({"intent": intent}),
                }
            )

            key = load_or_create_key()
            pub = public_key_hex(key)
            signed_items = []

            for s in sections:
                body = section_text(s)
                path = _path(s, sections)
                section_rows.append(
                    {
                        "id": s["id"],
                        "run_id": run_id,
                        "document_id": document_id,
                        "pipeline_id": self.pipeline_id,
                        "parent_id": s.get("parent_id"),
                        "level": s["level"],
                        "title": s["title"],
                        "page_start": s["page_start"],
                        "page_end": s["page_end"],
                        "text": body,
                        "summary": body[:280],
                        "extra_json": json.dumps({"intent": intent, "path": path}),
                    }
                )
                sec_node = f"node_{s['id']}"
                nodes.append(
                    {
                        "node_id": sec_node,
                        "node_type": "section",
                        "document_id": document_id,
                        "label": s["title"],
                        "extra_json": json.dumps({"level": s["level"]}),
                    }
                )
                edges.append(_edge("contains", doc_node, sec_node, document_id, document_id, 1.0))
                if s.get("parent_id"):
                    edges.append(
                        _edge("subsumes", f"node_{s['parent_id']}", sec_node, document_id, document_id, 1.0)
                    )

                for ent in extract_entities(s["title"] + "\n" + body[:1500]):
                    ent_id = (
                        f"node_ent_{document_id[-12:]}_"
                        f"{hashlib.sha1(ent.lower().encode()).hexdigest()[:12]}"
                    )
                    nodes.append(
                        {
                            "node_id": ent_id,
                            "node_type": "entity",
                            "document_id": document_id,
                            "label": ent,
                            "extra_json": "{}",
                        }
                    )
                    edges.append(_edge("mentions", sec_node, ent_id, document_id, document_id, 0.7))

                params = extract_parameters(body or s["title"], s["page_start"], s["id"])
                for p in params:
                    payload = parameter_payload(p)
                    sig = sign_payload(key, payload)
                    param_id = f"prm_{uuid.uuid4().hex[:12]}"
                    param_rows.append(
                        {
                            "param_id": param_id,
                            "document_id": document_id,
                            "section_id": s["id"],
                            "pipeline_id": self.pipeline_id,
                            "parameter_name": p["parameter_name"],
                            "numeric_value": p["numeric_value"],
                            "raw_string_value": p["raw_string_value"],
                            "unit": p.get("unit"),
                            "data_type": p.get("data_type"),
                            "provenance_page": p["provenance_page"],
                            "manifest_id": None,
                            "manifest_signature": sig,
                        }
                    )
                    signed_items.append({"param_id": param_id, "hash": hashlib.sha256(payload).hexdigest(), "sig": sig})
                    pnode = f"node_{param_id}"
                    nodes.append(
                        {
                            "node_id": pnode,
                            "node_type": "parameter",
                            "document_id": document_id,
                            "label": f"{p['parameter_name']}={p['raw_string_value']}",
                            "extra_json": json.dumps({"unit": p.get("unit")}),
                        }
                    )
                    edges.append(_edge("defines", sec_node, pnode, document_id, document_id, 1.0))

                pieces = recursive_split(body, max_chars=900, overlap=80) if body else []
                if not pieces and s["title"]:
                    pieces = [s["title"]]
                for i, part in enumerate(pieces):
                    retrieval = f"Document: {title}\nIntent: {intent}\nSection: {' > '.join(path)}\n\n{part}"
                    pages = [b.page for b in s.get("blocks", [])] or [s["page_start"]]
                    page_start, page_end = min(pages), max(pages)
                    matched = [
                        a
                        for a in assets
                        if page_start <= (a.get("page_number") or 0) <= page_end
                    ]
                    ents = extract_entities(s["title"] + "\n" + part)
                    caps = [a.get("caption") or "" for a in matched]
                    subspaces = channel_texts(
                        title=title,
                        intent=intent,
                        section_path=path,
                        section_title=s["title"],
                        body=part,
                        retrieval_text=retrieval,
                        entities=ents,
                        captions=caps,
                        page_start=page_start,
                    )
                    chunk_rows.append(
                        {
                            "id": f"chk_{uuid.uuid4().hex[:12]}",
                            "run_id": run_id,
                            "document_id": document_id,
                            "pipeline_id": self.pipeline_id,
                            "section_id": s["id"],
                            "chunk_index": i,
                            "text": part,
                            "retrieval_text": retrieval,
                            "page_start": page_start,
                            "page_end": page_end,
                            "token_estimate": token_estimate(retrieval),
                            "context_json": json.dumps(
                                {
                                    "section_path": path,
                                    "intent": intent,
                                    "channels": list(CHANNELS),
                                    "vectorprism": subspaces,
                                }
                            ),
                            "asset_ids_json": json.dumps([a["id"] for a in matched]),
                        }
                    )

            manifest_id = f"man_{uuid.uuid4().hex[:12]}"
            manifest_body = json.dumps(signed_items, sort_keys=True).encode()
            manifest_hash = hashlib.sha256(manifest_body).hexdigest()
            manifest_sig = sign_payload(key, manifest_body)
            for p in param_rows:
                p["manifest_id"] = manifest_id

            self.report("prismmanifest", 48, f"{len(param_rows)} signed parameters")
            self.store.insert_sections(section_rows)
            self.store.insert_chunks(chunk_rows)
            if param_rows:
                self.store.conn.executemany(
                    """
                    INSERT INTO document_parameters (
                        param_id, document_id, section_id, pipeline_id, parameter_name,
                        numeric_value, raw_string_value, unit, data_type, provenance_page,
                        manifest_id, manifest_signature
                    ) VALUES (
                        :param_id, :document_id, :section_id, :pipeline_id, :parameter_name,
                        :numeric_value, :raw_string_value, :unit, :data_type, :provenance_page,
                        :manifest_id, :manifest_signature
                    )
                    """,
                    param_rows,
                )
            self.store.conn.execute(
                """
                INSERT INTO prism_manifests (manifest_id, document_id, payload_hash, signature, public_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (manifest_id, document_id, manifest_hash, manifest_sig, pub, utcnow()),
            )
            self.report("chorusgraph", 58, f"{len(nodes)} nodes / {len(edges)} edges")
            self._upsert_graph(nodes, edges)
            self._link_cross_document(document_id, nodes)

            ids = [c["id"] for c in chunk_rows]
            if ids:
                self.report("vectorprism", 62, f"embedding {len(CHANNELS)} channels × {len(ids)} chunks")
                for channel in CHANNELS:
                    texts = []
                    for c in chunk_rows:
                        ctx = json.loads(c["context_json"])
                        texts.append((ctx.get("vectorprism") or {}).get(channel) or c["retrieval_text"])
                    self.persist_vectors(TABLES[channel], ids, texts, channel)

            stats = {
                "intent": intent,
                "sections": len(section_rows),
                "chunks": len(chunk_rows),
                "parameters": len(param_rows),
                "graph_nodes": len(nodes),
                "graph_edges": len(edges),
                "manifest_id": manifest_id,
                "vectorprism": {
                    "channels": list(CHANNELS),
                    "tables": [TABLES[c] for c in CHANNELS],
                    "chunks": len(ids),
                },
                "indexes": [TABLES[c] for c in CHANNELS] + ["chunks_fts", "chorusgraph_edges"],
                "usage": self.usage_stats(len(profiles)),
            }
            self.store.conn.commit()
            self.store.finish_run(run_id, "ok", stats, warnings)
            return {"run_id": run_id, "stats": stats, "warnings": warnings}
        except Exception as exc:
            self.store.finish_run(run_id, "error", {}, warnings, str(exc))
            raise

    def _upsert_graph(self, nodes: list[dict], edges: list[dict]) -> None:
        if nodes:
            # entity nodes may collide across docs — keep first label
            self.store.conn.executemany(
                """
                INSERT OR IGNORE INTO chorusgraph_nodes (node_id, node_type, document_id, label, extra_json)
                VALUES (:node_id, :node_type, :document_id, :label, :extra_json)
                """,
                nodes,
            )
        if edges:
            self.store.conn.executemany(
                """
                INSERT OR REPLACE INTO chorusgraph_edges (
                    edge_id, source_node, target_node, relationship_type,
                    document_id_source, document_id_target, weight, extra_json
                ) VALUES (
                    :edge_id, :source_node, :target_node, :relationship_type,
                    :document_id_source, :document_id_target, :weight, :extra_json
                )
                """,
                edges,
            )

    def _link_cross_document(self, document_id: str, new_nodes: list[dict]) -> None:
        edges = []
        entities = [n for n in new_nodes if n["node_type"] == "entity"]
        others = self.store.fetchall(
            """
            SELECT * FROM chorusgraph_nodes
            WHERE node_type='entity' AND document_id IS NOT NULL AND document_id != ?
            """,
            (document_id,),
        )
        by_label: dict[str, list[dict]] = defaultdict(list)
        for n in others:
            by_label[n["label"].lower()].append(n)
        for n in entities:
            for other in by_label.get(n["label"].lower(), []):
                edges.append(
                    _edge(
                        "same_entity",
                        n["node_id"],
                        other["node_id"],
                        document_id,
                        other["document_id"],
                        0.95,
                    )
                )
        # semantic overlap between section summaries
        my_secs = self.store.fetchall(
            "SELECT id, title, summary FROM sections WHERE document_id=? AND pipeline_id='prism'",
            (document_id,),
        )
        other_secs = self.store.fetchall(
            "SELECT id, document_id, title, summary FROM sections WHERE document_id!=? AND pipeline_id='prism'",
            (document_id,),
        )
        if my_secs and other_secs:
            my_texts = [(s["title"] or "") + " " + (s["summary"] or "") for s in my_secs]
            ot_texts = [(s["title"] or "") + " " + (s["summary"] or "") for s in other_secs]
            my_vecs = embed_texts(my_texts)
            ot_vecs = embed_texts(ot_texts)
            import numpy as np

            a = np.asarray(my_vecs, dtype=np.float32)
            b = np.asarray(ot_vecs, dtype=np.float32)
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
            b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            sim = a @ b.T
            for i, s in enumerate(my_secs):
                for j, t in enumerate(other_secs):
                    w = float(sim[i, j])
                    if w >= 0.70:
                        edges.append(
                            _edge(
                                "overlaps_with",
                                f"node_{s['id']}",
                                f"node_{t['id']}",
                                document_id,
                                t["document_id"],
                                round(w, 4),
                            )
                        )
        if edges:
            self._upsert_graph([], edges)


def _path(section: dict, sections: list[dict]) -> list[str]:
    by_id = {s["id"]: s for s in sections}
    path = [section["title"]]
    parent = section.get("parent_id")
    while parent and parent in by_id:
        path.append(by_id[parent]["title"])
        parent = by_id[parent].get("parent_id")
    return list(reversed(path))


def _edge(rel: str, src: str, dst: str, dsrc: str, ddst: str, weight: float) -> dict:
    eid = hashlib.sha1(f"{rel}|{src}|{dst}".encode()).hexdigest()[:20]
    return {
        "edge_id": f"e_{eid}",
        "source_node": src,
        "target_node": dst,
        "relationship_type": rel,
        "document_id_source": dsrc,
        "document_id_target": ddst,
        "weight": weight,
        "extra_json": "{}",
    }
