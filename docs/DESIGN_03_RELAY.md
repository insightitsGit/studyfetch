# Master Design Document 3: Relay (page-routed document intelligence)

## 1. Executive summary

Relay is the **cost-aware, assignment-shaped** pipeline: reconstruct structure that downstream AI can trust, spend multimodal/OCR budget only on pages that already failed, and keep provenance boring and complete.

It is not LangGraph-as-product (that is Baseline) and not zero-trust GraphRAG (that is Prism). Relay’s thesis: **most PDF pages are digitally extracted text; treating every page as a vision problem is how demos get expensive and irreproducible.**

## 2. Technical stack

| Layer | Choice |
|---|---|
| Extract | PyMuPDF blocks + font histogram + `find_tables` (shared ingest) |
| Page router | Deterministic labels: `digital_text`, `scanned`, `low_text`, `table_heavy`, `figure_heavy`, `mixed` |
| OCR | Tesseract **only** on `scanned` / `low_text` |
| LLM | Optional (`OPENAI_API_KEY`); default path is extractive summaries, zero API calls |
| Chunking | Leaf-section chunks; split only if `>1600` chars |
| Index | `vec_relay` + shared `chunks_fts` |
| Assets | First-class figures/tables with ids on chunks, binaries in Blob |

## 3. Pipeline

```
shared ingest (hash, pages, figures, tables)
        │
        ▼
per-page features (chars, image coverage, tables, density)
        │
        ├── digital_text / mixed / table_heavy / figure_heavy → keep PyMuPDF
        └── scanned / low_text → OCR that page only; else emit unusable stub
        │
        ▼
column reorder → header/footer strip → heading stack → section tree
        │
        ▼
section-aware chunks + context prefix + asset_ids → vec_relay
```

**Routing rule:** a multimodal model is a fallback for failed pages, not the default. In this timebox the fallback is Tesseract (local, free). A week-two swap is “one GPT-4o vision call per failed page,” still not per document.

## 4. What constitutes a chunk

**One leaf section** if it fits; otherwise paragraph/window splits that never cross a heading.

Each chunk stores:

```
Document: {title}
Section: {H1 > H2 > …}
Pages: {start}-{end}

{body}
```

That prefix is the entire parent context we duplicate. We do **not** concatenate sibling sections or the rest of the chapter.

Hierarchy is a tree (`parent_id`, `level`) plus a flat `sections` list. Pages stay on every section, chunk, figure, and table.

Figures and tables are **assets**, not pasted into `text`. Chunks that share a page range carry `asset_ids`. Captions are the retrievable string; pixels stay behind `/api/assets/{id}`.

## 5. What downstream retrieval should include

Include: `retrieval_text` (already prefixed), `section_path`, page range, `asset_ids`, quality blob (`page_methods`, boilerplate count), `contains_math` on the section.

Exclude: running headers, full-document text, raw image bytes, overlapping copies of the previous chunk.

## 6. Schema this pipeline writes

Shared tables only, plus **`vec_relay`**. Page `label` is updated so the UI can show the router. No manifests, no graph (Relay can *consume* Prism’s graph in a future mix; it does not own it).

## 7. Failure modes

| Mode | Honest behavior |
|---|---|
| Same font size for titles and body | Heading stack degrades; we still page-range the leftover “Document” section |
| Multi-column footnotes | Column cluster can interleave a sidebar |
| Table-as-image | Labeled `figure_heavy` / `scanned`; not a fake grid |
| Math as glyphs | `contains_math` flag; we do not reconstruct LaTeX |
| Repeated real titles in the header band | May be stripped as boilerplate |
| Unusable page | Warning in `pipeline_runs.warnings_json`; page is not silently dropped |

## 8. Tradeoffs vs the other two

- Vs **Baseline:** same extractors, but chunks prefer **whole sections** over sliding windows, and the router is page-typed (table/figure/scan) rather than a single density bit. No LangGraph — the graph would not change the decisions.
- Vs **Prism:** no crypto, no six-channel VectorPrism, no ChorusGraph. Relay is what you ship first for a study-product RAG. Prism is what you add when digits and cross-doc identity are the product.
- **Reproducibility over cleverness.** Cache by SHA-256; same bytes → same document id; re-run replaces that pipeline’s rows only.

## 9. How the bake-off scores this stack

Same graded rubric as the other two (nDCG@5, MRR, P@5, distractors, 75/25 retrieval/structure). Relay is usually even with Baseline on retrieval. Structure can show the same running-header leak as Baseline. Page labels (`digital_text` / `scanned` / …) are in the UI and in `pages`; they do not add bake-off points. Relay does not write bonus edges.

## 10. Representation (assignment questions)

| Question | Answer | How |
|---|---|---|
| Useful for downstream AI? | Yes — assignment-shaped study RAG | One leaf section per chunk (split only if long) plus `asset_ids` and page-router `label` (`digital_text` / `table_heavy` / `scanned` / …). A study product can cite the figure via `/api/assets/{id}` instead of pasting pixels. |
| Traceable to the original document? | Yes | `document_id` → filename / sha256, `section_id`, page range, `page_label`, `asset_ids`. Compare retrieval prints the router label next to the page number. |
| Context without unnecessary duplication? | Yes | Prefix is `Document / Section / Pages` only. The body is that leaf section — we do not slide across headings or concatenate siblings. Figures and tables are ids, not inlined markdown copies of every grid. |

Compare retrieval attaches this card as `design` so Relay is not a restyle of Baseline: leaf chunk, page label, asset count.

## 11. With another week

Vision fallback on `low_text` with a hard call budget; equation crops; heading F1 / table-exact eval; background queue for 200-page textbooks.
