from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.extract.pdf_common import (
    TextBlock,
    build_sections,
    detect_boilerplate,
    extract_blocks,
    ocr_page,
    open_pdf,
    page_profiles,
    pdf_metadata,
    recursive_split,
    reorder_page_blocks,
    section_text,
    strip_boilerplate,
)
from app.pipelines.base import Pipeline, new_run, token_estimate


class DocumentState(TypedDict):
    document_id: str
    run_id: str
    pdf_path: str
    title: str
    metadata: dict
    profiles: list
    widths: dict
    heights: dict
    blocks: list
    boilerplate: list
    sections: list
    chunks: list
    warnings: list
    ocr_pages: list
    stats: dict


SCAN_CHAR_THRESHOLD = 80
SCAN_DENSITY_THRESHOLD = 0.08


def _extract_node(state: DocumentState) -> DocumentState:
    doc = open_pdf(state["pdf_path"])
    try:
        meta = pdf_metadata(doc)
        title = meta.get("title") or state.get("title") or "Untitled"
        profiles = page_profiles(doc, state["pdf_path"])
        widths = {p.page_number: p.width for p in profiles}
        heights = {p.page_number: p.height for p in profiles}
        blocks = extract_blocks(doc)
        blocks = reorder_page_blocks(blocks, widths)
    finally:
        doc.close()
    return {
        **state,
        "title": title,
        "metadata": meta,
        "profiles": [p.__dict__ for p in profiles],
        "widths": widths,
        "heights": heights,
        "blocks": blocks,
        "warnings": list(state.get("warnings") or []),
    }


def _boilerplate_node(state: DocumentState) -> DocumentState:
    blocks: list[TextBlock] = state["blocks"]
    junk = detect_boilerplate(blocks, state["heights"], len(state["profiles"]))
    kept, removed = strip_boilerplate(blocks, junk, state["heights"])
    warnings = list(state["warnings"])
    if removed:
        warnings.append(f"stripped {len(removed)} repeating header/footer blocks")
    return {**state, "blocks": kept, "boilerplate": removed, "warnings": warnings}


def _quality_node(state: DocumentState) -> DocumentState:
    return state


def _route_ocr(state: DocumentState) -> str:
    for p in state["profiles"]:
        if p["char_count"] < SCAN_CHAR_THRESHOLD or p["text_density"] < SCAN_DENSITY_THRESHOLD:
            return "ocr"
    return "structure"


def _ocr_node(state: DocumentState) -> DocumentState:
    doc = open_pdf(state["pdf_path"])
    warnings = list(state["warnings"])
    ocr_pages = []
    existing_pages = {b.page for b in state["blocks"]}
    try:
        for p in state["profiles"]:
            if p["char_count"] >= SCAN_CHAR_THRESHOLD and p["text_density"] >= SCAN_DENSITY_THRESHOLD:
                continue
            page = doc[p["page_number"] - 1]
            text, method = ocr_page(page)
            ocr_pages.append(p["page_number"])
            if not text:
                warnings.append(f"page {p['page_number']}: OCR produced no text ({method})")
                continue
            if p["page_number"] in existing_pages:
                warnings.append(f"page {p['page_number']}: replaced sparse extract with {method}")
                state["blocks"] = [b for b in state["blocks"] if b.page != p["page_number"]]
            state["blocks"].append(
                TextBlock(
                    page=p["page_number"],
                    text=text,
                    x0=0,
                    y0=0,
                    x1=state["widths"][p["page_number"]],
                    y1=state["heights"][p["page_number"]],
                    font_size=11,
                    font_name="ocr",
                    bold=False,
                )
            )
            warnings.append(f"page {p['page_number']}: routed to OCR ({method})")
    finally:
        doc.close()
    state["blocks"] = reorder_page_blocks(state["blocks"], state["widths"])
    return {**state, "warnings": warnings, "ocr_pages": ocr_pages}


def _structure_node(state: DocumentState) -> DocumentState:
    sections = build_sections(state["blocks"], state["title"])
    return {**state, "sections": sections}


def _chunk_node(state: DocumentState) -> DocumentState:
    chunks: list[dict[str, Any]] = []
    for section in state["sections"]:
        path = _section_path(section, state["sections"])
        header = " > ".join(path)
        body = section_text(section)
        if not body.strip():
            continue
        parts = recursive_split(body, max_chars=1100, overlap=160)
        for i, part in enumerate(parts):
            retrieval = f"{header}\n\n{part}"
            pages = [b.page for b in section.get("blocks", [])] or [section["page_start"]]
            chunks.append(
                {
                    "id": f"chk_{uuid.uuid4().hex[:12]}",
                    "section_id": section["id"],
                    "chunk_index": i,
                    "text": part,
                    "retrieval_text": retrieval,
                    "page_start": min(pages),
                    "page_end": max(pages),
                    "token_estimate": token_estimate(retrieval),
                    "context_json": json.dumps({"section_path": path, "inherited_header": header}),
                    "asset_ids_json": "[]",
                }
            )
    return {**state, "chunks": chunks}


def _section_path(section: dict, sections: list[dict]) -> list[str]:
    by_id = {s["id"]: s for s in sections}
    path = [section["title"]]
    parent = section.get("parent_id")
    while parent and parent in by_id:
        path.append(by_id[parent]["title"])
        parent = by_id[parent].get("parent_id")
    return list(reversed(path))


def compile_graph():
    graph = StateGraph(DocumentState)
    graph.add_node("extract_page_blocks", _extract_node)
    graph.add_node("strip_running_headers", _boilerplate_node)
    graph.add_node("quality_check", _quality_node)
    graph.add_node("ocr_fallback", _ocr_node)
    graph.add_node("build_structure", _structure_node)
    graph.add_node("chunk_sections", _chunk_node)
    graph.set_entry_point("extract_page_blocks")
    graph.add_edge("extract_page_blocks", "strip_running_headers")
    graph.add_edge("strip_running_headers", "quality_check")
    graph.add_conditional_edges(
        "quality_check", _route_ocr, {"ocr": "ocr_fallback", "structure": "build_structure"}
    )
    graph.add_edge("ocr_fallback", "build_structure")
    graph.add_edge("build_structure", "chunk_sections")
    graph.add_edge("chunk_sections", END)
    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = compile_graph()
    return _GRAPH


class BaselinePipeline(Pipeline):
    pipeline_id = "baseline"
    display_name = "Standard Baseline (LangGraph)"

    def run(self, document_id: str, pdf_path: Path) -> dict[str, Any]:
        run_id = new_run(self.store, self.pipeline_id, document_id)
        doc_row = self.store.fetchone("SELECT * FROM documents WHERE id=?", (document_id,))
        self.begin_usage()
        try:
            self.report("langgraph extract", 10, "coordinate blocks + column reorder")
            result = get_graph().invoke(
                {
                    "document_id": document_id,
                    "run_id": run_id,
                    "pdf_path": str(pdf_path),
                    "title": (doc_row or {}).get("title") or "",
                    "metadata": {},
                    "profiles": [],
                    "widths": {},
                    "heights": {},
                    "blocks": [],
                    "boilerplate": [],
                    "sections": [],
                    "chunks": [],
                    "warnings": [],
                    "ocr_pages": [],
                    "stats": {},
                }
            )
            section_rows = []
            for s in result["sections"]:
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
                        "text": section_text(s),
                        "summary": "",
                        "extra_json": json.dumps({"block_count": len(s.get("blocks") or [])}),
                    }
                )
            self.report("persist sections", 55, f"{len(section_rows)} sections")
            self.store.insert_sections(section_rows)
            chunk_rows = [
                {
                    **c,
                    "run_id": run_id,
                    "document_id": document_id,
                    "pipeline_id": self.pipeline_id,
                }
                for c in result["chunks"]
            ]
            self.store.insert_chunks(chunk_rows)
            self.report("index chunks", 68, f"{len(chunk_rows)} chunks")
            self.persist_vectors(
                "vec_baseline",
                [c["id"] for c in chunk_rows],
                [c["retrieval_text"] for c in chunk_rows],
                "default",
            )
            stats = {
                "sections": len(section_rows),
                "chunks": len(chunk_rows),
                "ocr_pages": result.get("ocr_pages") or [],
                "boilerplate_removed": len(result.get("boilerplate") or []),
                "index": "vec_baseline + chunks_fts",
                "usage": self.usage_stats((doc_row or {}).get("page_count") or 0),
            }
            self.store.finish_run(run_id, "ok", stats, result.get("warnings") or [])
            return {"run_id": run_id, "stats": stats, "warnings": result.get("warnings") or []}
        except Exception as exc:
            self.store.finish_run(run_id, "error", {}, [], str(exc))
            raise
