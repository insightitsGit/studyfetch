from __future__ import annotations

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

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

MAX_LEAF_CHARS = 1600


def _path(section: dict, sections: list[dict]) -> list[str]:
    by_id = {s["id"]: s for s in sections}
    path = [section["title"]]
    parent = section.get("parent_id")
    while parent and parent in by_id:
        path.append(by_id[parent]["title"])
        parent = by_id[parent].get("parent_id")
    return list(reversed(path))


class RelayPipeline(Pipeline):
    """Cost-aware page router: digital parse by default, OCR only on failed pages."""

    pipeline_id = "relay"
    display_name = "Relay (page-routed document intelligence)"

    def run(self, document_id: str, pdf_path: Path) -> dict[str, Any]:
        run_id = new_run(self.store, self.pipeline_id, document_id)
        warnings: list[str] = []
        methods: dict[str, int] = defaultdict(int)
        self.begin_usage()
        try:
            self.report("open pdf", 8, pdf_path.name)
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

                self.report("page router", 25, f"{len(profiles)} pages")
                # Route each page; OCR only scanned / unusable pages
                kept: list[TextBlock] = [b for b in blocks]
                pages_present = {b.page for b in kept}
                for p in profiles:
                    if p.label in {"scanned", "low_text"}:
                        text, method = ocr_page(doc[p.page_number - 1])
                        methods[method or "ocr"] += 1
                        if text:
                            kept = [b for b in kept if b.page != p.page_number]
                            kept.append(
                                TextBlock(
                                    page=p.page_number,
                                    text=text,
                                    x0=0,
                                    y0=0,
                                    x1=p.width,
                                    y1=p.height,
                                    font_size=11,
                                    font_name="ocr",
                                    bold=False,
                                )
                            )
                            warnings.append(
                                f"page {p.page_number}: {p.label} → {method} ({len(text)} chars)"
                            )
                        else:
                            warnings.append(
                                f"page {p.page_number}: unusable ({p.label}, {method})"
                            )
                    else:
                        methods["pymupdf"] += 1
                        if p.label == "table_heavy":
                            warnings.append(f"page {p.page_number}: table-heavy, kept grid extract")
                        if p.label == "figure_heavy":
                            warnings.append(f"page {p.page_number}: figure-heavy, assets linked not inlined")
                if not pages_present:
                    pass
            finally:
                doc.close()

            self.report("section tree", 50, "headings + boilerplate strip")
            blocks = reorder_page_blocks(kept, widths)
            junk = detect_boilerplate(blocks, heights, len(profiles))
            blocks, removed = strip_boilerplate(blocks, junk, heights)
            if removed:
                warnings.append(f"stripped {len(removed)} repeating headers/footers")

            sections = build_sections(blocks, title)
            assets = self.store.fetchall(
                "SELECT * FROM assets WHERE document_id=?", (document_id,)
            )
            assets_by_page: dict[int, list[dict]] = defaultdict(list)
            for a in assets:
                assets_by_page[a["page_number"]].append(a)

            section_rows = []
            chunk_rows = []
            for s in sections:
                body = section_text(s)
                path = _path(s, sections)
                pages = list(range(s["page_start"], s["page_end"] + 1))
                linked = []
                for pg in pages:
                    linked.extend(assets_by_page.get(pg, []))
                asset_ids = [a["id"] for a in linked]
                summary = _extractive_summary(body)
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
                        "summary": summary,
                        "extra_json": json.dumps(
                            {
                                "path": path,
                                "asset_ids": asset_ids,
                                "contains_math": _looks_like_math(body),
                            }
                        ),
                    }
                )

                parts = _section_chunks(body)
                if not parts and (s["title"] or linked):
                    parts = [s["title"] or "(empty section)"]
                for i, part in enumerate(parts):
                    prefix = (
                        f"Document: {title}\n"
                        f"Section: {' > '.join(path)}\n"
                        f"Pages: {s['page_start']}-{s['page_end']}"
                    )
                    retrieval = f"{prefix}\n\n{part}"
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
                            "page_start": s["page_start"],
                            "page_end": s["page_end"],
                            "token_estimate": token_estimate(retrieval),
                            "context_json": json.dumps(
                                {
                                    "section_path": path,
                                    "context_prefix": prefix,
                                    "quality": {
                                        "page_methods": dict(methods),
                                        "boilerplate_stripped": len(removed),
                                    },
                                }
                            ),
                            "asset_ids_json": json.dumps(asset_ids),
                        }
                    )

            self.report("persist chunks", 68, f"{len(section_rows)} sections / {len(chunk_rows)} chunks")
            self.store.insert_sections(section_rows)
            self.store.insert_chunks(chunk_rows)
            self.persist_vectors(
                "vec_relay",
                [c["id"] for c in chunk_rows],
                [c["retrieval_text"] for c in chunk_rows],
                "default",
            )

            # Persist page labels used by this run
            for p in profiles:
                self.store.conn.execute(
                    "UPDATE pages SET label=? WHERE document_id=? AND page_number=?",
                    (p.label, document_id, p.page_number),
                )

            stats = {
                "sections": len(section_rows),
                "chunks": len(chunk_rows),
                "page_methods": dict(methods),
                "assets_linked": sum(len(json.loads(c["asset_ids_json"])) > 0 for c in chunk_rows),
                "index": "vec_relay + chunks_fts",
                "usage": self.usage_stats(len(profiles)),
            }
            self.store.conn.commit()
            self.store.finish_run(run_id, "ok", stats, warnings)
            return {"run_id": run_id, "stats": stats, "warnings": warnings}
        except Exception as exc:
            self.store.finish_run(run_id, "error", {}, warnings, str(exc))
            raise


def _section_chunks(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= MAX_LEAF_CHARS:
        return [text]
    return recursive_split(text, max_chars=MAX_LEAF_CHARS, overlap=120)


def _extractive_summary(text: str, max_sents: int = 3) -> str:
    if not text:
        return ""
    parts = [s.strip() for s in text.replace("?", ".").split(".") if len(s.strip()) > 40]
    if not parts:
        return text[:240]
    return ". ".join(parts[:max_sents])[:400]


def _looks_like_math(text: str) -> bool:
    return bool(text) and any(tok in text for tok in ("∑", "∫", "√", "≈", "≤", "≥", "^2", "_{", "\\frac"))
