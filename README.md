# Studyfetch Document Intelligence

Three pipelines, one Docker image, one SQLite file (`intelligence.db`). The point is three **indexes** on the same PDFs so you can compare them.

**Live URL:** [https://studyfetch.graysky-ae27c2bc.eastus.azurecontainerapps.io/](https://studyfetch.graysky-ae27c2bc.eastus.azurecontainerapps.io/)  
Azure Container Apps (eastus). Seeds are indexed on first boot. No auth. Container disk is ephemeral — a new replica re-seeds; uploaded PDFs do not persist.

| Pipeline | Design | Indexes |
|---|---|---|
| **Baseline** | [DESIGN_01](docs/DESIGN_01_BASELINE.md) | `vec_baseline` + FTS5 |
| **Prism** | [DESIGN_02](docs/DESIGN_02_PRISM.md) | VectorPrism **6ch** (`semantic`, `structural`, `title`, `entity`, `numeric`, `caption`) + FTS5 + ChorusGraph |
| **Relay** | [DESIGN_03](docs/DESIGN_03_RELAY.md) | `vec_relay` + FTS5 |

Seeds load on first boot: academic paper, Nexus-24 datasheet, and field note AN-24-07 (same world, different numbers — the VectorPrism demo). Upload or Remove PDFs anytime.

## Architecture

```
PDF → shared ingest (hash, pages, figures, tables, local or Azure blob)
        ├── Baseline / LangGraph     → vec_baseline
        ├── Prism / Cortex+Graph+VP  → six vec_prism_* + signed parameters
        └── Relay / page router      → vec_relay
        → intelligence.db (sqlite-vec + FTS5) → FastAPI workbench
```

Shared ingest catalogs the file once (no chunks, no vectors, no signatures). Classifier: score ≥ 3 → Prism (heavy); else Relay (light). Baseline is never auto-selected — it is the bake-off control. Each pipeline writes its own runs, sections, chunks, and vectors.

## Why three stacks

- **Baseline** — control. LangGraph step machine (extract → maybe OCR → structure → chunk). Recursive windows under a heading. One embedding channel. Not Graph RAG.
- **Relay** — assignment-shaped path. Page labels (`digital_text` / `scanned` / …). Tesseract only on failed pages. Leaf-section chunks with a context prefix and `asset_ids`. No per-page vision model.
- **Prism** — heavy / bonus. Intent router, ChorusGraph (cross-doc `overlaps_with` + `same_entity`), Ed25519-signed metrics, **VectorPrism** (six intent-weighted subspaces), PrismShield (flags invented same-unit drift; does not rewrite prose).

Embeddings: local **BAAI/bge-small-en-v1.5** (384-d, FastEmbed). OpenAI unused unless you set a key.

## Chunking, hierarchy, provenance

| | Baseline | Prism | Relay |
|---|---|---|---|
| Chunk | ~1100c under a heading (overlap 160) | ~900c section window (overlap 80) | leaf section; split only if >1600c |
| Hierarchy | font / numbering stack | same tree + graph nodes | same tree |
| Provenance | `page_start/end`, `section_id` | + signed page on each metric | + page label, `asset_ids` |
| Retrieval text | heading path + body | title + intent + path + body | Document / Section / Pages + body |

Downstream JSON is the contract: outline, chunks (`retrieval_text` + context), assets, parameters, graph edges.

## Bonus (cross-document)

Vector search (including VectorPrism) finds chunks similar to a **question**. The bonus is collection analysis with **no** question: after each Prism run, section title+summary cosine ≥ 0.70 → `overlaps_with`; exact capitalized labels → `same_entity`. Chunks stay per PDF; the **edge** is the shared map.

Workbench **Score** tab lists related pairs, why they matched, document-level overlap, and sections unique to one PDF. Ask walks those edges only when **Search library** is on.

## Run locally

```bash
cp .env.example .env
docker compose up --build
# or (PowerShell):  $env:DATA_DIR=".\data"; $env:PYTHONPATH="."; uvicorn app.main:app --reload --port 8001
```

- Docker Compose: [http://127.0.0.1:8000](http://127.0.0.1:8000)  
- Local uvicorn (as used in this repo): [http://127.0.0.1:8001](http://127.0.0.1:8001)

First start downloads the embed model and indexes seeds.

## Models / cost

| Call | When |
|---|---|
| FastEmbed | every chunk + query (local, $0) |
| Tesseract | scanned / low-text pages only |
| OpenAI / vision | off by default |
| Ed25519 | Prism parameters only |

**Score** builds probes from the current library (extracted parameters, unique terms, shared terms) — not a gold script for the seed PDFs. Graded bake-off: nDCG@5, MRR, P@5, provenance, distractors (75% retrieval / 25% structure). VectorPrism and ChorusGraph are listed, not extra points.

## Failure modes and tradeoffs

Headings fail on one-font slides. Image-only tables become figures. Entity nodes are capitalization, not NER. `overlaps_with` can link generic Introductions. Empty OCR is a warning, never a silent drop.

We built **three comparable indexes**, not one mega-pipeline. Baseline is the control. Relay spends routing so a textbook is not 40 vision calls. Prism spends six vector tables + a graph to protect digits and surface cross-doc meaning.

Deliberately not built: LayoutLM, per-page GPT-4o, auth, a marketing UI, RAGAS, graph drawing.

## Deliverables

| Item | Status |
|---|---|
| **Live URL** | [https://studyfetch.graysky-ae27c2bc.eastus.azurecontainerapps.io/](https://studyfetch.graysky-ae27c2bc.eastus.azurecontainerapps.io/) |
| **Repository** | https://github.com/insightitsGit/studyfetch — `docker compose up --build` or uvicorn above |
| **README** | This page: architecture, pipeline, models, schema, chunking, provenance, failures, tradeoffs |
| **5-minute Loom** | Record separately. Walk the live app and spend most of it on *why* (structure, routing, chunking, provenance, cost), not UI chrome |

## Another week

A scanned seed so OCR is visible; persist `/data` on Azure Files; graph drawing; gold heading F1 on a larger set; vision fallback with a hard budget.

```
app/pipelines/     baseline.py · prism.py · relay.py
app/vectorprism.py six-channel retrieve
app/bonus.py       cross-document report
app/db/schema.sql  shared intelligence.db
static/            workbench (Ask · Outline · Evidence · JSON · Score · Designs)
docs/              full designs
```
