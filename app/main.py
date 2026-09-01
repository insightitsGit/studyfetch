from __future__ import annotations

import json
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.db.store import get_store
from app.embeddings import warmup
from app.classify import classify_route_from_pages
from app.pipelines import PIPELINES, REGISTRY, get_pipeline, ingest_pdf, resolve_pdf_path
from app.progress import ProgressReporter, bind_progress, jobs
from app.retrieve import retrieve
from app.benchmark import run_benchmark, seed_queries
from app.bonus import analyze_collection
from app.seed import seed_corpus
from app.storage.blobs import blob_store

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

DESIGN_CATALOG = [
    {
        "id": "baseline",
        "file": "DESIGN_01_BASELINE.md",
        "title": "Design 1 — Standard Baseline (LangGraph)",
        "pipeline_id": "baseline",
        "summary": "LangGraph control. One vec + FTS5. Path-prefixed windows. Traceable pages/section ids. No graph.",
    },
    {
        "id": "prism",
        "file": "DESIGN_02_PRISM.md",
        "title": "Design 2 — Prism Stack (GraphRAG + Zero-Trust)",
        "pipeline_id": "prism",
        "summary": "6ch VectorPrism + ChorusGraph + Ed25519 manifest. PrismShield is a query-time gate (verified / unsigned / drifted), not an ingest step.",
    },
    {
        "id": "relay",
        "file": "DESIGN_03_RELAY.md",
        "title": "Design 3 — Relay (page-routed document intelligence)",
        "pipeline_id": "relay",
        "summary": "Page router. Leaf sections + asset ids + page labels. Prefix Document/Section/Pages. No graph.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    store = get_store()
    seed_queries(store)
    docs = seed_corpus(store)
    try:
        warmup()
    except Exception as exc:
        print(f"[startup] embedding warmup failed: {exc}")
    existing = store.fetchone("SELECT id FROM pipeline_runs WHERE status='ok' LIMIT 1")
    if not existing:
        print("[startup] indexing seed documents through baseline, prism, and relay…")
        for doc in docs:
            pdf_path = resolve_pdf_path(store, doc["id"])
            for name in PIPELINES:
                try:
                    get_pipeline(name, store).run(doc["id"], pdf_path)
                    print(f"[startup] {name} finished {doc['filename']}")
                except Exception as exc:
                    print(f"[startup] {name} failed on {doc['filename']}: {exc}")
    yield
    store.close()


app = FastAPI(title="Studyfetch Document Intelligence", lifespan=lifespan)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunBody(BaseModel):
    document_id: str
    pipelines: list[str] | None = None


class QueryBody(BaseModel):
    query: str
    pipeline_id: str
    k: int = 6
    intent: str | None = None
    apply_shield: bool = True
    document_id: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    store = get_store()
    return {
        "ok": True,
        "db": str(store.path),
        "pipelines": list(PIPELINES),
        "indexes": store.index_stats(),
        "azure_blob": bool(settings.azure_storage_connection_string),
    }


@app.get("/api/pipelines")
def list_pipelines():
    return [
        {
            "id": name,
            "name": cls.display_name,
            "indexes": {
                "baseline": ["vec_baseline", "chunks_fts"],
                "prism": [
                    "vec_prism_semantic",
                    "vec_prism_structural",
                    "vec_prism_title",
                    "vec_prism_entity",
                    "vec_prism_numeric",
                    "vec_prism_caption",
                    "chunks_fts",
                    "chorusgraph_edges",
                ],
                "relay": ["vec_relay", "chunks_fts"],
            }[name],
        }
        for name, cls in REGISTRY.items()
    ]


def _with_route(store, doc: dict) -> dict:
    meta = doc.get("metadata")
    if isinstance(doc.get("metadata_json"), str) and not meta:
        try:
            meta = json.loads(doc["metadata_json"])
        except json.JSONDecodeError:
            meta = {}
        doc["metadata"] = meta
    meta = meta or {}
    route = (meta or {}).get("route")
    if not route:
        pages = store.fetchall(
            "SELECT label, features_json FROM pages WHERE document_id=?",
            (doc["id"],),
        )
        assets = store.fetchall(
            "SELECT asset_type FROM assets WHERE document_id=?",
            (doc["id"],),
        )
        route = classify_route_from_pages(
            title=doc.get("title") or "",
            filename=doc.get("filename") or "",
            pages=pages,
            assets=assets,
        )
        meta = {**(meta or {}), "route": route}
        store.conn.execute(
            "UPDATE documents SET metadata_json=? WHERE id=?",
            (json.dumps(meta), doc["id"]),
        )
        store.conn.commit()
        doc["metadata"] = meta
    doc["route"] = route
    return doc


@app.get("/api/documents")
def list_documents():
    store = get_store()
    return [_with_route(store, d) for d in store.list_documents()]


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    store = get_store()
    doc = ingest_pdf(store, file.filename, data)
    if isinstance(doc.get("metadata_json"), str):
        try:
            doc["metadata"] = json.loads(doc["metadata_json"])
        except json.JSONDecodeError:
            doc["metadata"] = {}
    return _with_route(store, doc)


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    store = get_store()
    if not store.delete_document(document_id):
        raise HTTPException(404, "document not found")
    return {"ok": True, "deleted": document_id}


def _execute_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    store = get_store()
    try:
        pdf_path = resolve_pdf_path(store, job.document_id)
        results = {}
        for pipe_state in job.pipelines:
            name = pipe_state.pipeline_id
            reporter = ProgressReporter(job, name)
            reporter.start()
            pipeline = get_pipeline(name, store)
            bind_progress(pipeline, reporter)
            try:
                results[name] = pipeline.run(job.document_id, pdf_path)
                reporter.finish(ok=True)
            except Exception as exc:
                reporter.finish(ok=False, error=str(exc))
                raise
        job.results = results
        job.status = "ok"
        job.stage = "complete"
        job.detail = " · ".join(
            f"{k}: {v.get('stats', {}).get('chunks', '?')} chunks" for k, v in results.items()
        )
        job.finished_at = __import__("time").time()
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.stage = "failed"
        job.detail = traceback.format_exc(limit=4)
        job.finished_at = __import__("time").time()


@app.post("/api/run")
def run_pipelines(body: RunBody):
    store = get_store()
    doc = store.fetchone("SELECT * FROM documents WHERE id=?", (body.document_id,))
    if not doc:
        raise HTTPException(404, "document not found")
    names = body.pipelines or list(PIPELINES)
    for name in names:
        if name not in REGISTRY:
            raise HTTPException(400, f"unknown pipeline {name}")
    job = jobs.create(body.document_id, names)
    thread = threading.Thread(target=_execute_job, args=(job.id,), daemon=True)
    thread.start()
    return job.snapshot()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.snapshot()


@app.get("/api/documents/{document_id}/structure")
def document_structure(document_id: str, pipeline_id: str = Query(...)):
    if pipeline_id not in REGISTRY:
        raise HTTPException(400, "unknown pipeline")
    payload = get_store().document_structure(document_id, pipeline_id)
    if not payload:
        raise HTTPException(404, "document not found")
    for key in ("document", "run"):
        if payload.get(key):
            _parse_json_fields(payload[key])
    for collection in ("sections", "chunks", "pages", "assets", "parameters", "graph_nodes", "graph_edges"):
        for row in payload.get(collection) or []:
            _parse_json_fields(row)
    return payload


@app.post("/api/query")
def query_index(body: QueryBody):
    if body.pipeline_id not in REGISTRY:
        raise HTTPException(400, "unknown pipeline")
    return retrieve(
        get_store(),
        body.query,
        body.pipeline_id,
        k=body.k,
        intent=body.intent,
        apply_shield=body.apply_shield,
        document_id=body.document_id,
    )


@app.post("/api/benchmark")
def benchmark():
    return run_benchmark(get_store())


@app.get("/api/designs")
def list_designs():
    return DESIGN_CATALOG


@app.get("/api/designs/{design_id}")
def get_design(design_id: str):
    item = next((d for d in DESIGN_CATALOG if d["id"] == design_id), None)
    if not item:
        raise HTTPException(404, "unknown design")
    path = DOCS_DIR / item["file"]
    if not path.exists():
        raise HTTPException(404, f"missing {item['file']}")
    return {**item, "markdown": path.read_text(encoding="utf-8")}


@app.get("/design-docs/{design_id}")
def download_design(design_id: str):
    item = next((d for d in DESIGN_CATALOG if d["id"] == design_id), None)
    if not item:
        raise HTTPException(404, "unknown design")
    path = DOCS_DIR / item["file"]
    if not path.exists():
        raise HTTPException(404, f"missing {item['file']}")
    return FileResponse(path, media_type="text/markdown; charset=utf-8", filename=item["file"])


@app.get("/api/indexes")
def indexes():
    return get_store().index_stats()


@app.get("/api/graph")
def graph():
    store = get_store()
    return {
        "nodes": store.fetchall("SELECT * FROM chorusgraph_nodes"),
        "edges": store.fetchall("SELECT * FROM chorusgraph_edges"),
    }


@app.get("/api/bonus")
def bonus_cross_document():
    return analyze_collection(get_store())


@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str):
    row = get_store().fetchone("SELECT * FROM assets WHERE id=?", (asset_id,))
    if not row:
        raise HTTPException(404, "asset not found")
    if row["asset_type"] == "table":
        return Response(row["extra_json"], media_type="application/json")
    if not row.get("blob_uri"):
        raise HTTPException(404, "no binary")
    data = blob_store.get(row["blob_uri"])
    return Response(data, media_type="image/png")


def _parse_json_fields(row: dict) -> None:
    for key, val in list(row.items()):
        if key.endswith("_json") and isinstance(val, str) and val:
            try:
                row[key[:-5]] = json.loads(val)
            except json.JSONDecodeError:
                pass
