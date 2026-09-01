# Studyfetch Document Intelligence

Three pipelines, one Docker image, one SQLite file (`intelligence.db`). The point is three **indexes** on the same PDFs so you can compare them.

| Pipeline | Design | Indexes |
|---|---|---|
| **Baseline** | [DESIGN_01](docs/DESIGN_01_BASELINE.md) | `vec_baseline` + FTS5 |
| **Prism** | [DESIGN_02](docs/DESIGN_02_PRISM.md) | VectorPrism **6ch** (`semantic`, `structural`, `title`, `entity`, `numeric`, `caption`) + FTS5 + ChorusGraph |
| **Relay** | [DESIGN_03](docs/DESIGN_03_RELAY.md) | `vec_relay` + FTS5 |

Seeds (academic paper + Nexus-24 datasheet) load on first boot. Upload or Remove PDFs anytime.

## Architecture

```
PDF → shared ingest (hash, pages, figures, tables, local or Azure blob)
        ├── Baseline / LangGraph     → vec_baseline
        ├── Prism / Cortex+Graph+VP  → six vec_prism_* + signed parameters
        └── Relay / page router      → vec_relay
        → intelligence.db (sqlite-vec + FTS5) → FastAPI workbench
```

Shared ingest runs once. Each pipeline writes its own runs, sections, chunks, and vectors.

## Why three stacks

- **Baseline** — control. Coordinate reorder, header strip, OCR only if a page looks empty, recursive chunks under a heading. One embedding channel.
- **Relay** — assignment-shaped path. Page labels (`digital_text` / `scanned` / …). Tesseract only on failed pages. Leaf-section chunks with a context prefix and `asset_ids`. No per-page vision model.
- **Prism** — heavy / bonus. Intent router, ChorusGraph (cross-doc `overlaps_with` + `same_entity`), Ed25519-signed metrics, **VectorPrism** (six intent-weighted subspaces), PrismShield (flags invented same-unit drift; does not rewrite prose).

Embeddings: local **BAAI/bge-small-en-v1.5** (384-d, FastEmbed). OpenAI unused unless you set a key.

## Chunking, hierarchy, provenance

| | Baseline | Prism | Relay |
|---|---|---|---|
| Chunk | ~1100c under a heading | ~900c section window | leaf section (split if long) |
| Hierarchy | font / numbering stack | same tree + graph nodes | same tree |
| Provenance | `page_start/end`, `section_id` | + signed page on each metric | + page label, `asset_ids` |
| Retrieval text | heading path + body | title + intent + path + body | Document / Section / Pages + body |

Downstream JSON is the contract: outline, chunks (`retrieval_text` + context), assets, parameters, graph edges.

## Bonus (cross-document)

Prism links sections across PDFs when title+summary cosine ≥ 0.70 (`overlaps_with`) or they share an entity label (`same_entity`). Workbench **Benchmark** shows related pairs, why they matched, document-level overlap, and sections unique to one PDF. Search-all-documents uses the same indexes. Bake-off query `q_overlap` scores that path: VectorPrism 6ch mix, cross-doc hit-at-1 (from `retrieval_text`), and a ChorusGraph expand check.

## Run

```bash
cp .env.example .env
docker compose up --build
# or:  set DATA_DIR=.\data && uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000 (or whatever host you already have — a local workbench URL is enough). First start downloads the embed model and indexes seeds.

Azure: same image. Mount `/data` (or set `AZURE_STORAGE_CONNECTION_STRING` for blobs). No auth.

## Models / cost

| Call | When |
|---|---|
| FastEmbed | every chunk + query (local, $0) |
| Tesseract | scanned / low-text pages only |
| OpenAI / vision | off by default |
| Ed25519 | Prism parameters only |

Benchmark reports call counts vs “send every page to a multimodal model.”

## Failure modes and tradeoffs

Headings fail on one-font slides. Image-only tables become figures. Entity nodes are capitalization, not NER. `overlaps_with` can link generic Introductions. Empty OCR is a warning, never a silent drop.

We built **three comparable indexes**, not one mega-pipeline. Baseline is the control. Relay spends routing so a textbook is not 40 vision calls. Prism spends six vector tables + a graph to protect digits and surface cross-doc meaning.

Deliberately not built: LayoutLM, per-page GPT-4o, auth, a marketing UI, a public cloud deploy.

## Deliverables

| Item | Status |
|---|---|
| **Live URL** | Local workbench is enough — open the host you already have (typically `http://127.0.0.1:8000` or `:8001`). Seeds are indexed on first boot. |
| **Repository** | This tree. `docker compose up --build` or the `uvicorn` command above. |
| **README** | This page: architecture, pipeline, models, schema, chunking, provenance, failures, tradeoffs. |
| **5-minute Loom** | Record separately. Walk the live app and spend most of it on *why* (structure, routing, chunking, provenance, cost), not UI chrome. |

## Another week

A scanned seed so OCR is visible; graph drawing; gold heading F1 on a larger set; vision fallback with a hard budget.

```
app/pipelines/     baseline.py · prism.py · relay.py
app/vectorprism.py six-channel retrieve
app/bonus.py       cross-document report
app/db/schema.sql  shared intelligence.db
static/            workbench
docs/              full designs
```
