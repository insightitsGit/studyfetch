from __future__ import annotations

from collections import Counter
from typing import Any

from app.extract.pdf_common import (
    PageProfile,
    TextBlock,
    classify_document_intent,
    extract_parameters,
    heading_level,
    infer_body_size,
)

# Light = Relay: digital prose, clear headings, few metrics. One index, OCR only if a page fails.
# Heavy = Prism: numbers/tables/scans/cross-doc. Dual vectors, graph, signed parameters.
# Baseline is the LangGraph control — never auto-selected; use it for bake-offs.


def heading_stats(blocks: list[TextBlock]) -> dict[str, Any]:
    body_size = infer_body_size(blocks)
    headings: list[dict[str, Any]] = []
    for b in blocks:
        level = heading_level(b, body_size)
        if level is None:
            continue
        headings.append(
            {
                "text": " ".join(b.text.split())[:120],
                "level": level,
                "font_size": round(b.font_size, 1),
                "bold": b.bold,
                "page": b.page,
            }
        )
    return {
        "body_font_size": body_size,
        "heading_count": len(headings),
        "numbered_headings": sum(1 for h in headings if h["text"][:1].isdigit()),
        "headings": headings[:20],
        "rule": (
            "A block is a title/heading if it is short (<160 chars), larger than the "
            "dominant body font (or bold + a bit larger, or numbered 1. / 1.1), "
            "and not a multi-line paragraph. Everything else is body."
        ),
    }


def classify_route(
    *,
    title: str,
    filename: str,
    profiles: list[PageProfile],
    blocks: list[TextBlock],
    table_count: int,
    figure_count: int,
) -> dict[str, Any]:
    text = "\n".join(b.text for b in blocks)
    intent = classify_document_intent(text, profiles, title)
    heads = heading_stats(blocks)
    scanned = sum(1 for p in profiles if p.label in {"scanned", "low_text"})
    table_pages = sum(1 for p in profiles if p.table_count > 0)
    params = extract_parameters(text[:8000], 1, None)
    pages = max(len(profiles), 1)

    score = 0
    reasons: list[str] = []

    if scanned / pages >= 0.25:
        score += 2
        reasons.append(f"{scanned}/{pages} pages look scanned or low-text")
    if table_count >= 2 or table_pages >= 2:
        score += 2
        reasons.append(f"{table_count} tables / {table_pages} table-heavy pages")
    if intent in {"financial", "technical"}:
        score += 2
        reasons.append(f"intent={intent} (metrics / datasheet language)")
    if len(params) >= 5:
        score += 2
        reasons.append(f"{len(params)} numeric parameters to bind")
    if figure_count >= 4:
        score += 1
        reasons.append(f"{figure_count} figures")
    if pages >= 30:
        score += 1
        reasons.append(f"{pages} pages — graph + dual index pays off")
    if heads["heading_count"] >= 3 and score == 0:
        reasons.append(
            f"{heads['heading_count']} font-size headings vs body {heads['body_font_size']}pt — enough for the light stack"
        )

    heavy = score >= 3
    tier = "heavy" if heavy else "light"
    pipeline = "prism" if heavy else "relay"
    return {
        "tier": tier,
        "pipeline": pipeline,
        "also_useful": ["baseline"] if not heavy else ["relay", "baseline"],
        "score": score,
        "intent": intent,
        "reasons": reasons or ["Clean digital prose — default to the light stack"],
        "features": {
            "pages": pages,
            "scanned_pages": scanned,
            "tables": table_count,
            "figures": figure_count,
            "parameters": len(params),
            "heading_count": heads["heading_count"],
            "body_font_size": heads["body_font_size"],
        },
        "heading": heads,
        "legend": {
            "light": "Relay — one embedding index, section chunks, OCR only on failed pages. Use for stories, essays, papers with digital text.",
            "heavy": "Prism — dual vectors, ChorusGraph, Ed25519-signed numbers, Shield. Use for datasheets, financials, scans, or cross-doc work.",
            "baseline": "LangGraph control (not auto-run). Same extractors, sliding-window chunks. For bake-offs only.",
        },
    }


def classify_route_from_pages(
    *,
    title: str,
    filename: str,
    pages: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Weaker route for docs ingested before the classifier existed."""
    labels = Counter((p.get("label") or "") for p in pages)
    scanned = labels.get("scanned", 0) + labels.get("low_text", 0)
    tables = sum(1 for a in assets if a.get("asset_type") == "table")
    figures = sum(1 for a in assets if a.get("asset_type") == "figure")
    n = max(len(pages), 1)
    score = 0
    reasons = []
    if scanned / n >= 0.25:
        score += 2
        reasons.append(f"{scanned}/{n} scanned/low-text pages")
    if tables >= 2:
        score += 2
        reasons.append(f"{tables} tables")
    blob = f"{title} {filename}".lower()
    if any(k in blob for k in ("datasheet", "revenue", "voltage", "fiscal", "q4")):
        score += 2
        reasons.append("title/filename looks financial or technical")
    if figures >= 4:
        score += 1
        reasons.append(f"{figures} figures")
    heavy = score >= 3
    return {
        "tier": "heavy" if heavy else "light",
        "pipeline": "prism" if heavy else "relay",
        "also_useful": ["baseline"],
        "score": score,
        "intent": "unknown",
        "reasons": reasons or ["Default light (Relay) — no expensive signals"],
        "features": {
            "pages": n,
            "scanned_pages": scanned,
            "tables": tables,
            "figures": figures,
        },
        "heading": {
            "rule": (
                "Titles vs paragraphs: dominant body font from long blocks; a block is a heading "
                "if short and larger/bold/numbered (1., 1.1, Chapter)."
            )
        },
        "legend": {
            "light": "Relay — cheap page-routed stack.",
            "heavy": "Prism — graph + signed metrics.",
            "baseline": "LangGraph control for comparisons.",
        },
        "inferred": True,
    }
