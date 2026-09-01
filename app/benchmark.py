from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any

from app.bonus import analyze_collection
from app.db.store import Store, utcnow
from app.pipelines.base import PIPELINES
from app.retrieve import retrieve
from app.usage import reset as usage_reset
from app.usage import snapshot as usage_snapshot

DEFAULT_QUERIES = [
    {
        "id": "q_voltage",
        "query": "What is the maximum operating voltage?",
        "intent": "parameter",
        "notes": "Exact max is 24 V. 26 V is the table neighbor. Little Prince is a distractor.",
        "must_contain": ["24 V"],
        "prefer_contain": ["Maximum Operating Voltage"],
        "must_not_be_only": ["26 V"],
        "filename_hint": "nexus24",
        "avoid_filename": "prince",
    },
    {
        "id": "q_typical",
        "query": "typical operating voltage from the ratings table",
        "intent": "parameter",
        "notes": "Typ column is 24 V. Ranking the Max 26 V cell first is a miss.",
        "must_contain": ["24"],
        "prefer_contain": ["Typ", "Operating Voltage"],
        "must_not_be_only": ["26 V"],
        "filename_hint": "nexus24",
        "avoid_filename": "prince",
    },
    {
        "id": "q_isolation",
        "query": "What is the isolation tolerance?",
        "intent": "parameter",
        "notes": "1500 V type-test, not the 24 V continuous rating.",
        "must_contain": ["1500 V"],
        "prefer_contain": ["Isolation"],
        "must_not_be_only": [],
        "filename_hint": "nexus24",
        "avoid_filename": "prince",
    },
    {
        "id": "q_encoder",
        "query": "How does the encoder-decoder attention stack work?",
        "intent": "academic",
        "notes": "Must land on the paper's 3.1 stack, not the datasheet firmware recap or the novel.",
        "must_contain": ["encoder"],
        "prefer_contain": ["3.1", "Decoder"],
        "must_not_be_only": [],
        "filename_hint": "attention",
        "avoid_filename": "prince",
    },
    {
        "id": "q_revenue",
        "query": "Q4 revenue and units sold",
        "intent": "financial",
        "notes": "Audited $1,042,500 and 875 units. Wrong PDF or a neighboring figure should drop nDCG.",
        "must_contain": ["1,042,500"],
        "prefer_contain": ["875", "Q4"],
        "must_not_be_only": [],
        "filename_hint": "nexus24",
        "avoid_filename": "prince",
    },
    {
        "id": "q_overlap",
        "query": "transformer attention used in industrial controllers",
        "intent": "academic",
        "notes": "Needs both seed PDFs. A novel-only or single-doc list is a partial credit.",
        "must_contain": ["attention"],
        "prefer_contain": ["controller", "encoder"],
        "must_not_be_only": [],
        "filename_hint": "",
        "require_files": ["attention", "nexus24"],
        "avoid_filename": "prince",
        "cross_doc": True,
    },
    {
        "id": "q_fox",
        "query": "Why does the fox ask to be tamed?",
        "intent": "academic",
        "notes": "Little Prince only. Seed datasheet/paper hits are distractors.",
        "must_contain": ["fox"],
        "prefer_contain": ["tame"],
        "must_not_be_only": [],
        "filename_hint": "prince",
        "avoid_filename": "nexus24",
    },
    {
        "id": "q_baobab",
        "query": "What is the danger of the baobabs?",
        "intent": "academic",
        "notes": "Novel-specific. Technical PDFs should not win this ranking.",
        "must_contain": ["baobab"],
        "prefer_contain": ["planet"],
        "must_not_be_only": [],
        "filename_hint": "prince",
        "avoid_filename": "attention",
    },
]

HEADING_GOLD = [
    (
        ("attention_routing", "attention"),
        ["Introduction", "Related Work", "Model Architecture", "Experiments", "Conclusion"],
    ),
    (
        ("nexus24", "datasheet"),
        ["Overview", "Electrical Ratings", "Commercial", "Firmware", "Safety"],
    ),
    (
        ("little_prince", "prince"),
        ["Chapter 1", "Chapter 5", "Chapter 21", "Chapter 27"],
    ),
]

BOILERPLATE_NEEDLES = (
    "confidential draft",
    "headers repeat",
    "studyfetch seed corpus",
)


def seed_queries(store: Store) -> None:
    ids = [q["id"] for q in DEFAULT_QUERIES]
    placeholders = ",".join("?" * len(ids))
    store.conn.execute(f"DELETE FROM benchmark_queries WHERE id NOT IN ({placeholders})", ids)
    for q in DEFAULT_QUERIES:
        store.conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_queries (id, query, intent, notes)
            VALUES (:id, :query, :intent, :notes)
            """,
            {"id": q["id"], "query": q["query"], "intent": q["intent"], "notes": q["notes"]},
        )
    store.conn.commit()


def _gold(query_id: str) -> dict[str, Any]:
    for q in DEFAULT_QUERIES:
        if q["id"] == query_id:
            return q
    return {}


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


def _grade_hit(hit: dict, gold: dict[str, Any]) -> tuple[float, str]:
    """Graded relevance 0–3. Distractors and neighbor digits score 0."""
    text = _hit_text(hit)
    text_l = text.lower()
    must = gold.get("must_contain") or []
    prefer = gold.get("prefer_contain") or []
    wrong = gold.get("must_not_be_only") or []
    hint = gold.get("filename_hint") or ""
    avoid = gold.get("avoid_filename") or ""
    must_hits = sum(1 for m in must if m.lower() in text_l)
    prefer_hits = sum(1 for p in prefer if p.lower() in text_l)

    if avoid and _hint_match(hit, avoid):
        return 0.0, f"distractor file ({avoid})"
    if wrong and any(w.lower() in text_l for w in wrong) and must_hits < max(len(must), 1):
        return 0.0, "neighbor metric without the exact value"
    if hint and not _hint_match(hit, hint):
        if must_hits == 0 and prefer_hits == 0:
            return 0.0, f"wrong document (wanted {hint})"
        return 0.6, f"cues present but wrong document (wanted {hint})"

    if must and must_hits == len(must) and prefer and prefer_hits == len(prefer):
        return 3.0, "right document + required string + section cues"
    if must and must_hits == len(must):
        extra = 0.4 * (prefer_hits / max(len(prefer), 1)) if prefer else 0.0
        return round(2.0 + extra, 2), "right document + required string"
    if must_hits or prefer_hits:
        partial = (must_hits / max(len(must), 1)) + 0.35 * (prefer_hits / max(len(prefer), 1))
        return round(min(1.4, partial), 2), "partial cue match"
    if hint and _hint_match(hit, hint):
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
    k = max(len(hits), 5)
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

    if gold.get("cross_doc") or gold.get("require_files"):
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
    required = gold.get("require_files") or []
    if required:
        found = {
            hint
            for hint in required
            if any(_hint_match(h, hint) for h in hits)
        }
        coverage = len(found) / len(required)
        extras *= coverage
        extra_notes.append(f"required docs {len(found)}/{len(required)}")
    if gold.get("avoid_filename") and hits:
        if _hint_match(hits[0], gold["avoid_filename"]):
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
    if gold.get("filename_hint"):
        add("right_document", _hint_match(top, gold["filename_hint"]) if hits else False, f"hint {gold['filename_hint']}")
    if gold.get("avoid_filename"):
        add("distractor_clean", distractors == 0, f"{distractors} {gold['avoid_filename']} hit(s) in top-k")
    if required or gold.get("cross_doc"):
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


def _heading_recall(filename: str, sections: list[dict]) -> dict[str, Any] | None:
    name = (filename or "").lower()
    gold = None
    for keys, headings in HEADING_GOLD:
        if any(k in name for k in keys):
            gold = headings
            break
    if not gold:
        return None
    blob = " ".join((s.get("title") or "") for s in sections).lower()
    found = [h for h in gold if h.lower() in blob]
    return {
        "expected": gold,
        "found": found,
        "recall": round(len(found) / len(gold), 3) if gold else 0,
    }


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
        if not sections and not chunks:
            continue
        levels = sorted({s["level"] for s in sections}) if sections else []
        leak = sum(
            1
            for c in chunks
            if any(n in (c.get("retrieval_text") or "").lower() for n in BOILERPLATE_NEEDLES)
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
        heading = _heading_recall(doc["filename"], sections)
        per_doc.append(
            {
                "document_id": doc["id"],
                "filename": doc["filename"],
                "sections": len(sections),
                "heading_levels": levels,
                "heading_recall": heading["recall"] if heading else None,
                "heading_found": heading["found"] if heading else [],
                "chunks": len(chunks),
                "provenance_coverage": round(
                    sum(1 for c in chunks if c.get("page_start")) / max(len(chunks), 1), 3
                ),
                "section_context_coverage": round(with_path / max(len(chunks), 1), 3),
                "boilerplate_leak_rate": round(leak / max(len(chunks), 1), 3),
                "chunks_with_assets": with_assets,
            }
        )
    recalls = [d["heading_recall"] for d in per_doc if d["heading_recall"] is not None]
    vp_channels = []
    if pipeline_id == "prism":
        vp_channels = [
            r["channel"]
            for r in store.fetchall(
                "SELECT DISTINCT channel FROM embeddings_meta WHERE pipeline_id='prism' ORDER BY channel"
            )
        ]
    avg_recall = (sum(recalls) / len(recalls)) if recalls else 0.0
    avg_leak = (
        sum(d["boilerplate_leak_rate"] for d in per_doc) / max(len(per_doc), 1) if per_doc else 0.0
    )
    avg_ctx = (
        sum(d["section_context_coverage"] for d in per_doc) / max(len(per_doc), 1) if per_doc else 0.0
    )
    structure_score = round(100 * (0.70 * avg_recall + 0.20 * (1 - avg_leak) + 0.10 * avg_ctx), 1)
    return {
        "pipeline_id": pipeline_id,
        "documents": per_doc,
        "vectorprism_channels": vp_channels,
        "avg_heading_recall": round(avg_recall, 3) if recalls else None,
        "avg_boilerplate_leak": round(avg_leak, 3) if per_doc else None,
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
    seed_queries(store)
    queries = store.fetchall("SELECT * FROM benchmark_queries")
    results = []
    retrieve_usage = []
    for q in queries:
        gold = _gold(q["id"])
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
        results.append({"query": {**q, **{k: gold.get(k) for k in ("must_contain", "notes")}}, "pipelines": per_pipeline})
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
        "rubric": {
            "document_understanding": (
                "Graded 0–3 per hit (right doc + exact string + section cues). "
                "Score = 40% nDCG@5 + 25% MRR + 15% P@5 + 10% provenance/path + 10% distractor/coverage. "
                "Overall = 75% retrieval + 25% structure. Capability flags (6ch, graph) are listed, not added."
            ),
            "representation": "page provenance + section path + no running headers on the top-3",
            "system_design": "LLM/vision/OCR/embed call counts and $ vs naïve per-page vision",
            "cost": "local FastEmbed + Tesseract = $0; OpenAI only if you set a key",
        },
        "cost": cost_rollup(store),
        "structure": structure,
        "leaderboard": leaderboard,
        "retrieve_usage": retrieve_usage,
        "bonus": analyze_collection(store),
        "results": results,
    }
