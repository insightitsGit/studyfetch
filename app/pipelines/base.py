from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.db.store import Store, utcnow
from app.embeddings import embed_texts
from app.usage import reset as usage_reset
from app.usage import snapshot as usage_snapshot
from app.extract.pdf_common import (
    extract_blocks,
    extract_images,
    extract_tables_plumber,
    extract_tables_pymupdf,
    attach_captions,
    open_pdf,
    page_profiles,
    pdf_metadata,
)
from app.classify import classify_route
from app.storage.blobs import blob_store, sha256_bytes

PIPELINES = ("baseline", "prism", "relay")


class Pipeline(ABC):
    pipeline_id: str
    display_name: str

    def __init__(self, store: Store):
        self.store = store
        self.report = lambda *_args, **_kwargs: None

    def begin_usage(self) -> None:
        usage_reset()

    def usage_stats(self, page_count: int = 0) -> dict:
        return usage_snapshot(page_count)

    @abstractmethod
    def run(self, document_id: str, pdf_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    def persist_vectors(self, table: str, chunk_ids: list[str], texts: list[str], channel: str) -> None:
        if not chunk_ids:
            return
        batch = 16
        vectors: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, batch):
            part = texts[i : i + batch]
            vectors.extend(embed_texts(part))
            done = min(total, i + len(part))
            self.report(
                f"embedding {channel}",
                70 + int(28 * done / max(total, 1)),
                f"{done}/{total} chunks → {table}",
            )
        self.store.insert_vectors(table, list(zip(chunk_ids, vectors)), self.pipeline_id, channel)


def ingest_pdf(store: Store, filename: str, data: bytes) -> dict[str, Any]:
    digest = sha256_bytes(data)
    existing = store.fetchone("SELECT * FROM documents WHERE sha256=?", (digest,))
    if existing:
        row = dict(existing)
        row["already_present"] = True
        return row

    document_id = f"doc_{digest[:16]}"
    blob_uri = blob_store.put(f"{document_id}/source.pdf", data, "application/pdf")
    pdf_path = _materialize(blob_uri, document_id)

    doc = open_pdf(str(pdf_path))
    try:
        meta = pdf_metadata(doc)
        title = meta.get("title") or Path(filename).stem.replace("_", " ")
        profiles = page_profiles(doc, str(pdf_path))
        blocks = extract_blocks(doc)
        figures = extract_images(doc, document_id)
        tables = extract_tables_plumber(str(pdf_path), document_id)
        if not tables:
            tables = extract_tables_pymupdf(doc, document_id)
        attach_captions(figures + tables, blocks)
    finally:
        doc.close()

    route = classify_route(
        title=title,
        filename=filename,
        profiles=profiles,
        blocks=blocks,
        table_count=len(tables),
        figure_count=len(figures),
    )
    meta = {**meta, "route": route}
    store.upsert_document(
        {
            "id": document_id,
            "filename": filename,
            "sha256": digest,
            "title": title,
            "page_count": len(profiles),
            "metadata_json": json.dumps(meta),
            "blob_uri": blob_uri,
            "created_at": utcnow(),
        }
    )
    store.replace_pages(
        document_id,
        [
            {
                "id": f"pg_{document_id}_{p.page_number}",
                "document_id": document_id,
                "page_number": p.page_number,
                "char_count": p.char_count,
                "image_coverage": p.image_coverage,
                "text_density": p.text_density,
                "label": p.label,
                "features_json": json.dumps(
                    {
                        "width": p.width,
                        "height": p.height,
                        "word_count": p.word_count,
                        "table_count": p.table_count,
                        "image_count": p.image_count,
                    }
                ),
                "text_preview": "",
            }
            for p in profiles
        ],
    )
    store.replace_assets(document_id, figures + tables)
    row = store.fetchone("SELECT * FROM documents WHERE id=?", (document_id,))
    if row:
        row["already_present"] = False
    return row


def resolve_pdf_path(store: Store, document_id: str) -> Path:
    doc = store.fetchone("SELECT * FROM documents WHERE id=?", (document_id,))
    if not doc:
        raise FileNotFoundError(document_id)
    return _materialize(doc["blob_uri"], document_id)


def _materialize(uri: str, document_id: str) -> Path:
    local = blob_store.local_path(uri)
    if local and local.exists():
        return local
    from app.config import settings

    dest = settings.upload_dir / f"{document_id}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob_store.get(uri))
    return dest


def new_run(store: Store, pipeline_id: str, document_id: str) -> str:
    store.delete_run_outputs(document_id, pipeline_id)
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    store.create_run(
        {
            "id": run_id,
            "pipeline_id": pipeline_id,
            "document_id": document_id,
            "status": "running",
            "started_at": utcnow(),
            "finished_at": None,
            "stats_json": "{}",
            "warnings_json": "[]",
            "error": None,
        }
    )
    return run_id


def token_estimate(text: str) -> int:
    return max(1, len(text.split()))
