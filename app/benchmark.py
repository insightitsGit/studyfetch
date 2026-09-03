from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter, defaultdict
from typing import Any

from app.bonus import analyze_collection
from app.db.store import Store, utcnow
from app.extract.pdf_common import extract_parameters
from app.pipelines.base import PIPELINES
from app.retrieve import retrieve
from app.usage import reset as usage_reset
from app.usage import snapshot as usage_snapshot

MAX_QUERIES = 12
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{3,}")
HEADING_LINE_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,/'&:-]{2,72}$"
)
STOP = {
    "this", "that", "with", "from", "have", "been", "were", "they", "them",
    "their", "what", "when", "where", "which", "into", "your", "about",
    "page", "pages", "figure", "table", "chapter", "section", "document",
    "confidential", "draft", "headers", "repeat", "studyfetch", "seed",
    "corpus", "introduction", "conclusion", "overview", "references",
    "abstract", "appendix", "contents", "index",
}

BOILERPLATE_NEEDLES = (
    "confidential draft",
    "headers repeat",
    "studyfetch seed corpus",
    "all rights reserved",
    "copyright",
)


def _ensure_gold_column(store: Store) -> None:
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(benchmark_queries)").fetchall()}
    if "gold_json" not in cols:
        store.conn.execute(
            "ALTER TABLE benchmark_queries ADD COLUMN gold_json TEXT NOT NULL DEFAULT '{}'"
        )
        store.conn.commit()


def corpus_from_store(store: Store) -> dict[str, Any]:
    docs = store.fetchall("SELECT id, filename, title, page_count FROM documents")
    pages = store.fetchall(
        "SELECT document_id, page_number, text_preview, label FROM pages ORDER BY document_id, page_number"
    )
    assets = store.fetchall("SELECT document_id, caption, extra_json FROM assets")
    return {"documents": docs, "pages": pages, "assets": assets}


def generate_probes(
    documents: list[dict],
    pages: list[dict],
    assets: list[dict] | None = None,
    *,
    max_queries: int = MAX_QUERIES,
) -> list[dict[str, Any]]:
    """Build graded probes from the current library — not a fixed seed script."""
    if not documents:
        return []
    assets = assets or []
    names = {d["id"]: (d.get("title") or d["filename"] or d["id"]) for d in documents}
    files = {d["id"]: d.get("filename") or d["id"] for d in documents}
    other_ids = lambda doc_id: [d["id"] for d in documents if d["id"] != doc_id]

    pages_by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        pages_by_doc[p["document_id"]].append(p)

    probes: list[dict[str, Any]] = []
    used_queries: set[str] = set()

    def add(probe: dict[str, Any]) -> None:
        q = re.sub(r"\s+", " ", (probe.get("query") or "")).strip()
        if not q or q.lower() in used_queries:
            return
        if len(probes) >= max_queries:
            return
        used_queries.add(q.lower())
        kind = probe.get("kind") or "probe"
        n = sum(1 for p in probes if p.get("kind") == kind) + 1
        probe["id"] = probe.get("id") or f"q_{kind}_{n}"
        probe["query"] = q
        probes.append(probe)

    _add_parameter_probes(add, documents, pages_by_doc, names, files, other_ids)
    _add_section_probes(add, pages_by_doc, names, files, other_ids)
    _add_term_probes(add, documents, pages_by_doc, names, files, other_ids)
    _add_cross_doc_probes(add, documents, pages_by_doc, names, files)
    _add_caption_probes(add, assets, names, files, other_ids)

    if not probes:
        for doc in documents:
            title = (doc.get("title") or doc.get("filename") or "").strip()
            if len(title) < 4:
                continue
            add(
                {
                    "kind": "document",
                    "intent": "academic",
                    "query": f"What is {title} about?",
                    "notes": f"Fallback probe from the document title of {files[doc['id']]}.",
                    "must_contain": [w for w in WORD_RE.findall(title) if w.lower() not in STOP][:2],
                    "prefer_contain": [title],
                    "must_not_be_only": [],
                    "document_id": doc["id"],
                    "avoid_document_ids": other_ids(doc["id"]),
                    "source": {"filename": files[doc["id"]], "page": 1},
                }
            )
    return probes[:max_queries]


def _add_parameter_probes(add, documents, pages_by_doc, names, files, other_ids) -> None:
    harvested: list[dict] = []
    for doc in documents:
        for page in pages_by_doc.get(doc["id"], []):
            for param in extract_parameters(page.get("text_preview") or "", page["page_number"], None):
                harvested.append({**param, "document_id": doc["id"]})
    named = [p for p in harvested if not str(p.get("parameter_name") or "").lower().startswith("metric_")]
    pool = named or harvested
    pool.sort(key=lambda p: (-_param_weight(p), p.get("parameter_name") or "", p.get("raw_string_value") or ""))

    seen_names: set[str] = set()
    added = 0
    for param in pool:
        if added >= 4:
            break
        name = (param.get("parameter_name") or "").strip()
        raw = (param.get("raw_string_value") or "").strip()
        if not name or not raw or len(name) < 3:
            continue
        key = name.lower()
        if key in seen_names:
            continue
        # Prefer facts that live in one document so the gold doc is unambiguous.
        owners = {p["document_id"] for p in harvested if (p.get("parameter_name") or "").lower() == key}
        if len(owners) != 1:
            continue
        seen_names.add(key)
        neighbors = [
            p.get("raw_string_value")
            for p in harvested
            if p["document_id"] == param["document_id"]
            and (p.get("unit") or "") == (param.get("unit") or "")
            and (p.get("raw_string_value") or "") != raw
        ]
        neighbors = [n for n in neighbors if n][:2]
        intent = "financial" if (param.get("unit") or "").upper() in {"USD", "$"} or raw.startswith("$") else "parameter"
        doc_id = param["document_id"]
        add(
            {
                "kind": "parameter",
                "intent": intent,
                "query": f"What is the {name}?",
                "notes": (
                    f"Auto-built from page {param.get('provenance_page')} of {files[doc_id]}. "
                    f"Expect {raw}."
                    + (f" Neighbor value {neighbors[0]} is a miss." if neighbors else "")
                ),
                "must_contain": [raw],
                "prefer_contain": [name],
                "must_not_be_only": neighbors,
                "document_id": doc_id,
                "avoid_document_ids": other_ids(doc_id),
                "source": {"filename": files[doc_id], "page": param.get("provenance_page"), "document": names[doc_id]},
            }
        )
        added += 1


def _param_weight(param: dict) -> int:
    name = (param.get("parameter_name") or "").lower()
    score = 0
    if not name.startswith("metric_"):
        score += 4
    if any(w in name for w in ("maximum", "minimum", "typical", "nominal", "peak", "revenue", "isolation", "voltage", "current", "timeout", "sold", "price")):
        score += 3
    if param.get("unit") in {"V", "A", "W", "USD", "$", "ms"}:
        score += 1
    return score


def _add_section_probes(add, pages_by_doc, names, files, other_ids) -> None:
    added = 0
    for doc_id, doc_pages in pages_by_doc.items():
        if added >= 2:
            break
        headings = []
        for page in doc_pages:
            for line in (page.get("text_preview") or "").splitlines():
                line = line.strip()
                if not HEADING_LINE_RE.match(line):
                    continue
                if line.lower() in STOP or len(line) < 6:
                    continue
                if any(n in line.lower() for n in BOILERPLATE_NEEDLES):
                    continue
                headings.append((line, page.get("page_number") or 1))
        # Prefer numbered / distinctive headings over generic "Introduction".
        headings.sort(key=lambda h: (0 if re.match(r"^\d", h[0]) else 1, -len(h[0])))
        picked = None
        for title, page in headings:
            words = [w for w in WORD_RE.findall(title) if w.lower() not in STOP]
            if len(words) < 1:
                continue
            picked = (title, page, words)
            break
        if not picked:
            continue
        title, page, words = picked
        add(
            {
                "kind": "section",
                "intent": "outline",
                "query": f"What does the section {title} cover?",
                "notes": f"Heading taken from page {page} of {files[doc_id]} (shared ingest, not a pipeline outline).",
                "must_contain": words[:2],
                "prefer_contain": [title],
                "must_not_be_only": [],
                "document_id": doc_id,
                "avoid_document_ids": other_ids(doc_id),
                "source": {"filename": files[doc_id], "page": page, "document": names[doc_id]},
            }
        )
        added += 1


def _doc_term_counts(pages_by_doc: dict[str, list[dict]]) -> dict[str, Counter]:
    counts: dict[str, Counter] = {}
    for doc_id, doc_pages in pages_by_doc.items():
        blob = " ".join((p.get("text_preview") or "") for p in doc_pages)
        counts[doc_id] = Counter(w.lower() for w in WORD_RE.findall(blob) if w.lower() not in STOP)
    return counts


def _add_term_probes(add, documents, pages_by_doc, names, files, other_ids) -> None:
    counts = _doc_term_counts(pages_by_doc)
    df = Counter()
    for c in counts.values():
        for term in c:
            df[term] += 1
    added = 0
    candidates = []
    for doc in documents:
        for term, tf in counts.get(doc["id"], {}).items():
            if df[term] != 1 or tf < 1 or len(term) < 5:
                continue
            original = _original_casing(pages_by_doc.get(doc["id"], []), term)
            candidates.append((-len(term), -tf, term, original, doc["id"]))
    candidates.sort()
    seen_docs: set[str] = set()
    for _, _, term, original, doc_id in candidates:
        if added >= 3:
            break
        if doc_id in seen_docs and added >= 1:
            continue
        seen_docs.add(doc_id)
        add(
            {
                "kind": "unique_term",
                "intent": "academic",
                "query": f"What does the document say about {original}?",
                "notes": f"Term “{original}” appears only in {files[doc_id]} among the current library.",
                "must_contain": [original],
                "prefer_contain": [],
                "must_not_be_only": [],
                "document_id": doc_id,
                "avoid_document_ids": other_ids(doc_id),
                "source": {"filename": files[doc_id], "document": names[doc_id]},
            }
        )
        added += 1


def _original_casing(pages: list[dict], term: str) -> str:
    for page in pages:
        for word in WORD_RE.findall(page.get("text_preview") or ""):
            if word.lower() == term:
                return word
    return term


def _add_cross_doc_probes(add, documents, pages_by_doc, names, files) -> None:
    if len(documents) < 2:
        return
    counts = _doc_term_counts(pages_by_doc)
    shared: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        for term, tf in counts.get(doc["id"], {}).items():
            if tf >= 1 and len(term) >= 6:
                shared[term].append(doc["id"])
    pairs = []
    for term, owners in shared.items():
        uniq = list(dict.fromkeys(owners))
        if len(uniq) < 2:
            continue
        pairs.append((term, uniq[:3]))
    pairs.sort(key=lambda x: (-len(x[0]), x[0]))
    added = 0
    used_docs: set[tuple[str, ...]] = set()
    for term, owners in pairs:
        if added >= 2:
            break
        key = tuple(sorted(owners))
        if key in used_docs:
            continue
        used_docs.add(key)
        original = _original_casing(pages_by_doc.get(owners[0], []), term)
        add(
            {
                "kind": "cross_document",
                "intent": "academic",
                "query": f"{original} discussed across the library",
                "notes": (
                    f"“{original}” appears in {len(owners)} documents. "
                    "A single-doc top-k is partial credit."
                ),
                "must_contain": [original],
                "prefer_contain": [names[o] for o in owners if names.get(o)][:2],
                "must_not_be_only": [],
                "document_id": "",
                "require_document_ids": owners,
                "avoid_document_ids": [d["id"] for d in documents if d["id"] not in owners],
                "cross_doc": True,
                "source": {"filenames": [files[o] for o in owners]},
            }
        )
        added += 1


def _add_caption_probes(add, assets, names, files, other_ids) -> None:
    added = 0
    for asset in assets:
        if added >= 1:
            break
        caption = (asset.get("caption") or "").strip()
        if len(caption) < 8:
            extra = asset.get("extra_json")
            if isinstance(extra, str) and extra:
                try:
                    extra = json.loads(extra)
                except json.JSONDecodeError:
                    extra = {}
            if isinstance(extra, dict):
                headers = extra.get("headers") or []
                caption = " ".join(str(h) for h in headers if h).strip()
        if len(caption) < 8:
            continue
        words = [w for w in WORD_RE.findall(caption) if w.lower() not in STOP]
        if not words:
            continue
        doc_id = asset["document_id"]
        add(
            {
                "kind": "caption",
                "intent": "technical",
                "query": f"Where is {caption[:80]} described?",
                "notes": f"Caption / table header from {files.get(doc_id, doc_id)}.",
                "must_contain": words[:2],
                "prefer_contain": [caption[:80]],
                "must_not_be_only": [],
                "document_id": doc_id,
                "avoid_document_ids": other_ids(doc_id),
                "source": {"filename": files.get(doc_id), "document": names.get(doc_id)},
            }
        )
        added += 1


def seed_queries(store: Store) -> list[dict[str, Any]]:
    _ensure_gold_column(store)
    corpus = corpus_from_store(store)
    probes = generate_probes(corpus["documents"], corpus["pages"], corpus["assets"])
    ids = [p["id"] for p in probes] or ["_none"]
    placeholders = ",".join("?" * len(ids))
    store.conn.execute(f"DELETE FROM benchmark_queries WHERE id NOT IN ({placeholders})", ids)
    for p in probes:
        store.conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_queries (id, query, intent, notes, gold_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (p["id"], p["query"], p.get("intent"), p.get("notes"), json.dumps(p)),
        )
    store.conn.commit()
    return probes


def _gold(query_row: dict[str, Any]) -> dict[str, Any]:
    raw = query_row.get("gold_json")
    if isinstance(raw, dict):
        return {**query_row, **raw}
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {**query_row, **parsed}
        except json.JSONDecodeError:
            pass
    return dict(query_row)


def _blob(hits: list[dict]) -> str:
    return "\n".join((h.get("retrieval_text") or h.get("text") or "") for h in hits).lower()


def _context(hit: dict) -> dict:
    ctx = hit.get("context")
    if isinstance(ctx, dict):
        return ctx
    raw = hit.get("context_json")
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _norm_hint(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _hit_text(hit: dict) -> str:
    return hit.get("retrieval_text") or hit.get("text") or ""


def _hit_file(hit: dict) -> str:
    return _norm_hint(" ".join([hit.get("filename") or "", hit.get("document_id") or "", _hit_text(hit)[:80]]))


def _hint_match(hit: dict, hint: str) -> bool:
    if not hint:
        return False
    return _norm_hint(hint) in _hit_file(hit)


def _hit_doc(hit: dict) -> str:
    return hit.get("document_id") or ""


def _is_target_doc(hit: dict, gold: dict[str, Any]) -> bool:
    target = gold.get("document_id") or ""
    if target:
        return _hit_doc(hit) == target
    hint = gold.get("filename_hint") or ""
    return _hint_match(hit, hint) if hint else False


def _is_distractor(hit: dict, gold: dict[str, Any]) -> bool:
    avoids = list(gold.get("avoid_document_ids") or [])
    if gold.get("avoid_document_id"):
        avoids.append(gold["avoid_document_id"])
    if _hit_doc(hit) and _hit_doc(hit) in {a for a in avoids if a}:
        return True
    avoid = gold.get("avoid_filename") or ""
    return bool(avoid and _hint_match(hit, avoid))


def _grade_hit(hit: dict, gold: dict[str, Any]) -> tuple[float, str]:
    """Graded relevance 0–3. Distractors and neighbor digits score 0."""
    text = _hit_text(hit)
    text_l = text.lower()
    must = gold.get("must_contain") or []
    prefer = gold.get("prefer_contain") or []
    wrong = gold.get("must_not_be_only") or []
    must_hits = sum(1 for m in must if m.lower() in text_l)
    prefer_hits = sum(1 for p in prefer if p.lower() in text_l)
    targeted = bool(gold.get("document_id") or gold.get("filename_hint"))

    if _is_distractor(hit, gold):
        label = gold.get("avoid_filename") or "other document"
        return 0.0, f"distractor file ({label})"
    if wrong and any(w.lower() in text_l for w in wrong) and must_hits < max(len(must), 1):
        return 0.0, "neighbor metric without the exact value"
    if targeted and not _is_target_doc(hit, gold):
        if must_hits == 0 and prefer_hits == 0:
            return 0.0, "wrong document"
        return 0.6, "cues present but wrong document"

    if must and must_hits == len(must) and prefer and prefer_hits == len(prefer):
        return 3.0, "right document + required string + section cues"
    if must and must_hits == len(must):
        extra = 0.4 * (prefer_hits / max(len(prefer), 1)) if prefer else 0.0
        return round(2.0 + extra, 2), "right document + required string"
    if must_hits or prefer_hits:
        partial = (must_hits / max(len(must), 1)) + 0.35 * (prefer_hits / max(len(prefer), 1))
        return round(min(1.4, partial), 2), "partial cue match"
    if targeted and _is_target_doc(hit, gold):
        return 0.4, "right document, missing required string"
    return 0.0, "off-topic"


def _ndcg(rels: list[float], ideal: list[float]) -> float:
    def dcg(values: list[float]) -> float:
        return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(values))

    denom = dcg(ideal)
    if denom <= 0:
        return 0.0
    return min(1.0, dcg(rels) / denom)


def _representation(hits: list[dict]) -> tuple[float, list[dict]]:
    if not hits:
        return 0.0, []
    sample = hits[:3]
    checks = []
    scores = []
    for h in sample:
        text = _hit_text(h)
        ctx = _context(h)
        path = ctx.get("section_path") or ctx.get("inherited_header") or ctx.get("context_prefix") or []
        if isinstance(path, str):
            path = [path]
        prov = 1.0 if h.get("page_start") else 0.0
        section = 1.0 if path or "section:" in text.lower() or ">" in text[:200] else 0.0
        clean = 0.0 if any(n in text.lower() for n in BOILERPLATE_NEEDLES) else 1.0
        scores.append((prov + section + clean) / 3)
    score = sum(scores) / len(scores)
    checks.append({"name": "representation", "pass": score >= 0.67, "detail": f"top-3 provenance/path/clean={score:.2f}"})
    return score, checks


def score_retrieval(gold: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    hits = payload.get("hits") or []
    grades = []
    rels = []
    for i, hit in enumerate(hits):
        rel, why = _grade_hit(hit, gold)
        rels.append(rel)
        grades.append(
            {
                "rank": i + 1,
                "rel": rel,
                "why": why,
                "filename": hit.get("filename"),
            }
        )
    while len(rels) < 5:
        rels.append(0.0)

    if gold.get("cross_doc") or gold.get("require_document_ids") or gold.get("require_files"):
        ideal = [3.0, 3.0, 2.0, 1.0, 0.0]
    else:
        ideal = [3.0, 2.0, 1.0, 0.0, 0.0]
    ndcg = _ndcg(rels[:5], ideal)
    first_good = next((i + 1 for i, r in enumerate(rels) if r >= 2.0), None)
    mrr = (1.0 / first_good) if first_good else 0.0
    precision = sum(1 for r in rels[:5] if r >= 2.0) / 5
    distractors = sum(1 for g in grades if "distractor" in (g.get("why") or ""))
    representation, rep_checks = _representation(hits)

    extras = 1.0
    extra_notes = []
    required_ids = list(gold.get("require_document_ids") or [])
    required_files = list(gold.get("require_files") or [])
    if required_ids:
        found = {hid for hid in required_ids if any(_hit_doc(h) == hid for h in hits)}
        extras *= len(found) / len(required_ids)
        extra_notes.append(f"required docs {len(found)}/{len(required_ids)}")
    elif required_files:
        found = {hint for hint in required_files if any(_hint_match(h, hint) for h in hits)}
        extras *= len(found) / len(required_files)
        extra_notes.append(f"required docs {len(found)}/{len(required_files)}")
    if (gold.get("avoid_document_ids") or gold.get("avoid_filename")) and hits:
        if _is_distractor(hits[0], gold):
            extras *= 0.4
            extra_notes.append("distractor at rank 1")
        elif distractors:
            extras *= max(0.55, 1 - 0.15 * distractors)
            extra_notes.append(f"{distractors} distractor(s) in top-k")

    score = round(
        100
        * (
            0.40 * ndcg
            + 0.25 * mrr
            + 0.15 * precision
            + 0.10 * representation
            + 0.10 * extras
        ),
        1,
    )

    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    top = hits[0] if hits else {}
    top_text = _hit_text(top)
    all_text = _blob(hits)
    must = gold.get("must_contain") or []
    prefer = gold.get("prefer_contain") or []
    add("exact_content", all(m.lower() in all_text for m in must) if must else bool(hits), "top-k contains " + ", ".join(must) if must else "has hits")
    add("hit_at_1", bool(hits) and rels[0] >= 2.0, grades[0]["why"] if grades else "no hits")
    if prefer:
        add("heading_or_context", any(p.lower() in all_text for p in prefer), "named section / nearby cue")
    wrong = gold.get("must_not_be_only") or []
    if wrong:
        unsafe = any(w.lower() in top_text.lower() for w in wrong) and not all(m.lower() in top_text.lower() for m in must)
        add("digit_safe", not unsafe, "did not substitute a neighboring number")
    checks.extend(rep_checks)
    if gold.get("document_id") or gold.get("filename_hint"):
        add("right_document", _is_target_doc(top, gold) if hits else False, "gold document")
    if gold.get("avoid_document_ids") or gold.get("avoid_filename"):
        add("distractor_clean", distractors == 0, f"{distractors} distractor hit(s) in top-k")
    if required_ids or required_files or gold.get("cross_doc"):
        docs = {h.get("document_id") for h in hits if h.get("document_id")}
        add("cross_document", len(docs) >= 2, f"{len(docs)} documents in top-k")
    if payload.get("pipeline_id") == "prism":
        vp = payload.get("vectorprism") or {}
        chans = vp.get("channels") or []
        add("vectorprism_6ch", len(chans) >= 6, "capability — not counted in the 100")
        if gold.get("cross_doc"):
            graph_hits = [h for h in hits if h.get("graph_edge") or "chorusgraph" in (h.get("channels") or [])]
            add("chorusgraph_related", bool(graph_hits), "capability — graph expand" if graph_hits else "no graph expand")

    add("ndcg", ndcg >= 0.7, f"nDCG@5={ndcg:.2f} (graded 0–3, not pass/fail)")
    add("mrr", mrr >= 0.5, f"MRR={mrr:.2f} (first hit with rel≥2)")
    add("precision", precision >= 0.4, f"P@5={precision:.2f} (rel≥2)")

    passed = sum(1 for c in checks if c["pass"])
    return {
        "score": score,
        "passed": passed,
        "total": len(checks),
        "ndcg": round(ndcg, 3),
        "mrr": round(mrr, 3),
        "precision": round(precision, 3),
        "representation": round(representation, 3),
        "extras": round(extras, 3),
        "grades": grades,
        "extra_notes": extra_notes,
        "checks": checks,
    }


def running_headers(pages: list[dict]) -> set[str]:
    """Lines that repeat across most pages — generic running headers, not a seed list."""
    if not pages:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        seen = set()
        for raw in (page.get("text_preview") or "").splitlines():
            line = re.sub(r"\s+", " ", raw).strip().lower()
            if 4 <= len(line) <= 80:
                seen.add(line)
        for line in seen:
            counts[line] += 1
    need = max(2, math.ceil(len(pages) * 0.5))
    headers = {line for line, n in counts.items() if n >= need}
    headers.update(BOILERPLATE_NEEDLES)
    return headers


def heading_integrity(sections: list[dict], page_count: int) -> float:
    if not sections:
        return 0.0
    titles = [(s.get("title") or "").strip() for s in sections]
    nonempty = sum(1 for t in titles if t and t.lower() not in {"untitled", "document", "page"})
    title_score = nonempty / len(titles)
    levels = {s.get("level") for s in sections}
    hierarchy = 1.0 if len(levels) >= 2 else (0.55 if len(sections) >= 3 else 0.35)
    covered = len({s.get("page_start") for s in sections if s.get("page_start")})
    cover = min(1.0, covered / max(page_count or 1, 1))
    unique_ratio = len({t.lower() for t in titles if t}) / max(len(titles), 1)
    return round(0.40 * title_score + 0.25 * hierarchy + 0.20 * cover + 0.15 * unique_ratio, 3)


def structure_quality(store: Store, pipeline_id: str) -> dict[str, Any]:
    docs = store.fetchall("SELECT id, filename, page_count FROM documents")
    per_doc = []
    for doc in docs:
        sections = store.fetchall(
            "SELECT title, level, page_start, page_end FROM sections WHERE document_id=? AND pipeline_id=?",
            (doc["id"], pipeline_id),
        )
        chunks = store.fetchall(
            "SELECT retrieval_text, page_start, context_json, asset_ids_json FROM chunks WHERE document_id=? AND pipeline_id=?",
            (doc["id"], pipeline_id),
        )
        pages = store.fetchall(
            "SELECT text_preview FROM pages WHERE document_id=?",
            (doc["id"],),
        )
        if not sections and not chunks:
            continue
        headers = running_headers(pages)
        leak = sum(
            1
            for c in chunks
            if any(n in (c.get("retrieval_text") or "").lower() for n in headers)
        )
        with_path = 0
        for c in chunks:
            try:
                ctx = json.loads(c["context_json"] or "{}")
            except json.JSONDecodeError:
                ctx = {}
            if ctx.get("section_path") or ctx.get("inherited_header") or ctx.get("context_prefix"):
                with_path += 1
        with_assets = 0
        for c in chunks:
            try:
                aids = json.loads(c["asset_ids_json"] or "[]")
            except json.JSONDecodeError:
                aids = []
            if aids:
                with_assets += 1
        integrity = heading_integrity(sections, doc.get("page_count") or 0)
        provenance = sum(1 for c in chunks if c.get("page_start")) / max(len(chunks), 1)
        ctx_cov = with_path / max(len(chunks), 1)
        leak_rate = leak / max(len(chunks), 1)
        per_doc.append(
            {
                "document_id": doc["id"],
                "filename": doc["filename"],
                "sections": len(sections),
                "heading_levels": sorted({s["level"] for s in sections}) if sections else [],
                "heading_integrity": integrity,
                "heading_recall": integrity,
                "heading_found": [s.get("title") for s in sections[:8] if s.get("title")],
                "chunks": len(chunks),
                "provenance_coverage": round(provenance, 3),
                "section_context_coverage": round(ctx_cov, 3),
                "boilerplate_leak_rate": round(leak_rate, 3),
                "chunks_with_assets": with_assets,
            }
        )
    vp_channels = []
    if pipeline_id == "prism":
        vp_channels = [
            r["channel"]
            for r in store.fetchall(
                "SELECT DISTINCT channel FROM embeddings_meta WHERE pipeline_id='prism' ORDER BY channel"
            )
        ]
    if per_doc:
        avg_integrity = sum(d["heading_integrity"] for d in per_doc) / len(per_doc)
        avg_leak = sum(d["boilerplate_leak_rate"] for d in per_doc) / len(per_doc)
        avg_ctx = sum(d["section_context_coverage"] for d in per_doc) / len(per_doc)
        avg_prov = sum(d["provenance_coverage"] for d in per_doc) / len(per_doc)
    else:
        avg_integrity = avg_leak = avg_ctx = avg_prov = 0.0
    structure_score = round(
        100 * (0.35 * avg_integrity + 0.25 * avg_prov + 0.20 * avg_ctx + 0.20 * (1 - avg_leak)),
        1,
    )
    return {
        "pipeline_id": pipeline_id,
        "documents": per_doc,
        "vectorprism_channels": vp_channels,
        "avg_heading_integrity": round(avg_integrity, 3) if per_doc else None,
        "avg_heading_recall": round(avg_integrity, 3) if per_doc else None,
        "avg_boilerplate_leak": round(avg_leak, 3) if per_doc else None,
        "avg_provenance": round(avg_prov, 3) if per_doc else None,
        "structure_score": structure_score,
    }


def _parse_stats(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def cost_rollup(store: Store) -> list[dict[str, Any]]:
    rows = []
    for pipeline_id in PIPELINES:
        runs = store.fetchall(
            """
            SELECT document_id, stats_json, started_at
            FROM pipeline_runs
            WHERE pipeline_id=? AND status='ok'
            ORDER BY started_at DESC
            """,
            (pipeline_id,),
        )
        seen: set[str] = set()
        latest = []
        for r in runs:
            if r["document_id"] in seen:
                continue
            seen.add(r["document_id"])
            latest.append(_parse_stats(r["stats_json"]))
        usage = {
            "llm_calls": 0,
            "vision_pages": 0,
            "ocr_pages": 0,
            "embed_docs": 0,
            "embed_tokens": 0,
            "ed25519_signs": 0,
            "estimated_usd_actual": 0.0,
            "estimated_usd_if_every_page_vision": 0.0,
        }
        for st in latest:
            u = st.get("usage") or {}
            for k in usage:
                usage[k] += u.get(k, 0) or 0
        rows.append(
            {
                "pipeline_id": pipeline_id,
                "runs": len(latest),
                "llm_calls": usage["llm_calls"],
                "vision_pages": usage["vision_pages"],
                "ocr_pages": usage["ocr_pages"],
                "embed_docs": usage["embed_docs"],
                "ed25519_signs": usage["ed25519_signs"],
                "estimated_usd_actual": round(usage["estimated_usd_actual"], 6),
                "estimated_usd_if_every_page_vision": round(
                    usage["estimated_usd_if_every_page_vision"], 4
                ),
                "usd_saved_vs_per_page_vision": round(
                    max(
                        0.0,
                        usage["estimated_usd_if_every_page_vision"] - usage["estimated_usd_actual"],
                    ),
                    4,
                ),
                "note": (
                    "LLM/vision = 0 by design (deterministic + local embed). "
                    if usage["llm_calls"] == 0 and usage["vision_pages"] == 0
                    else "External model calls were recorded."
                ),
            }
        )
    return rows


def run_benchmark(store: Store, k: int = 5) -> dict[str, Any]:
    queries = seed_queries(store)
    results = []
    retrieve_usage = []
    for q in queries:
        gold = _gold(q)
        per_pipeline = []
        for pipeline_id in PIPELINES:
            usage_reset()
            payload = retrieve(
                store,
                q["query"],
                pipeline_id,
                k=k,
                intent=q.get("intent"),
                document_id=None,
            )
            quality = score_retrieval(gold, payload)
            retrieve_cost = usage_snapshot(0)
            retrieve_usage.append({"query_id": q["id"], "pipeline_id": pipeline_id, **retrieve_cost})
            rid = f"br_{uuid.uuid4().hex[:12]}"
            store.conn.execute(
                """
                INSERT INTO benchmark_results (
                    id, query_id, pipeline_id, index_name, latency_ms, hit_count, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    q["id"],
                    pipeline_id,
                    payload["index_name"],
                    payload["latency_ms"],
                    len(payload["hits"]),
                    json.dumps(
                        {
                            "quality": quality,
                            "usage": retrieve_cost,
                            "hits": [
                                {
                                    "chunk_id": h.get("chunk_id"),
                                    "document_id": h.get("document_id"),
                                    "page_start": h.get("page_start"),
                                    "score": h.get("score"),
                                    "preview": (h.get("retrieval_text") or "")[:240],
                                    "channels": h.get("channels"),
                                    "shield": h.get("shield"),
                                    "verified_parameters": h.get("verified_parameters"),
                                }
                                for h in payload["hits"]
                            ],
                        }
                    ),
                    utcnow(),
                ),
            )
            per_pipeline.append({**payload, "quality": quality, "usage": retrieve_cost})
        shown = {
            k: gold.get(k)
            for k in (
                "must_contain",
                "notes",
                "kind",
                "source",
                "document_id",
                "require_document_ids",
            )
        }
        results.append({"query": {**q, **shown}, "pipelines": per_pipeline})
    store.conn.commit()

    structure = [structure_quality(store, p) for p in PIPELINES]
    struct_by = {s["pipeline_id"]: s for s in structure}
    quality_by_pipeline: dict[str, list[dict]] = {p: [] for p in PIPELINES}
    for block in results:
        for p in block["pipelines"]:
            quality_by_pipeline[p["pipeline_id"]].append(p["quality"])
    leaderboard = []
    for p in PIPELINES:
        qs = quality_by_pipeline[p]
        retrieval = round(sum(q["score"] for q in qs) / len(qs), 1) if qs else 0.0
        ndcg = round(sum(q.get("ndcg") or 0 for q in qs) / len(qs), 3) if qs else 0.0
        mrr = round(sum(q.get("mrr") or 0 for q in qs) / len(qs), 3) if qs else 0.0
        precision = round(sum(q.get("precision") or 0 for q in qs) / len(qs), 3) if qs else 0.0
        struct = struct_by.get(p) or {}
        structure_score = float(struct.get("structure_score") or 0)
        overall = round(0.75 * retrieval + 0.25 * structure_score, 1)
        leaderboard.append(
            {
                "pipeline_id": p,
                "overall": overall,
                "avg_retrieval_quality": retrieval,
                "structure_score": structure_score,
                "ndcg": ndcg,
                "mrr": mrr,
                "precision": precision,
            }
        )
    leaderboard.sort(key=lambda r: r["overall"], reverse=True)

    return {
        "queries": len(queries),
        "pipelines": list(PIPELINES),
        "index_stats": store.index_stats(),
        "probe_plan": [
            {
                "id": q["id"],
                "kind": q.get("kind"),
                "query": q["query"],
                "intent": q.get("intent"),
                "notes": q.get("notes"),
                "source": q.get("source"),
            }
            for q in queries
        ],
        "rubric": {
            "document_understanding": (
                "Probes are generated from the current library (page text, unique terms, shared terms). "
                "Score = 40% nDCG@5 + 25% MRR + 15% P@5 + 10% provenance/path + 10% distractor/coverage. "
                "Overall = 75% retrieval + 25% structure. Capability flags (6ch, graph) are listed, not added."
            ),
            "representation": "page provenance + section path + no running headers on the top-3",
            "system_design": "LLM/vision/OCR/embed call counts and $ vs naïve per-page vision",
            "cost": "local FastEmbed + Tesseract = $0; OpenAI only if you set a key",
            "structure": (
                "Heading integrity (titles, hierarchy, page coverage) + provenance + section path "
                "+ generic running-header leak. Not a gold outline for a named PDF."
            ),
        },
        "cost": cost_rollup(store),
        "structure": structure,
        "leaderboard": leaderboard,
        "retrieve_usage": retrieve_usage,
        "bonus": analyze_collection(store),
        "results": results,
    }
