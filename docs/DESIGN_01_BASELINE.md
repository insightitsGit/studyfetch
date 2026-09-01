# Master Design Document 1: Standard Baseline Stack

## 1. Executive summary

The Standard Baseline Stack is a **resilient, interpretable control pipeline**. It is the system we measure the others against: deterministic layout repair, a LangGraph state machine for recovery, recursive character chunks with inherited section headers, and a single `sqlite-vec` channel plus FTS5.

It does **not** try to be a knowledge graph, a crypto gate, or a cost-aware multimodal router. Those are Prism and Relay. Baseline exists so a reviewer can answer: “if we only did the obvious industrial RAG pipeline well, what would retrieval look like?”

## 2. Technical stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** `StateGraph` | Conditional OCR routing and a recoverable `DocumentState` |
| Text | **PyMuPDF** coordinate blocks `(x0,y0,x1,y1)` | Fast, reproducible, font sizes for headings |
| Tables | **pdfplumber** `find_tables` / `extract_tables` | Explicit table boundaries, not “text that looks tabular” |
| OCR fallback | Tesseract, only when density fails | No per-page vision API |
| Chunking | Recursive split + sliding overlap + inherited header | Industry default; easy to explain |
| Index | `vec_baseline` (`float[384]`) + shared `chunks_fts` | One semantic channel, one lexical channel |
| Store | Shared `intelligence.db` | Same documents as Prism/Relay |

Embedding model (shared): `BAAI/bge-small-en-v1.5` via FastEmbed.

## 3. Pipeline (LangGraph)

```
extract → boilerplate → quality ──low density──► ocr ──► structure → chunk → persist
                              └──ok──────────► structure ─────────────┘
```

1. **Ingestion & profiling.** PDF hashed, stored once (Azure Blob or local `file://`). Pages labeled with density / image coverage for the shared `pages` table.
2. **Coordinate extraction.** Blocks reordered by clustering `x0` (columns) then sorting `y0`.
3. **Boilerplate strip.** Strings that repeat in the top 8% / bottom 8% of **≥30%** of pages are purged (page numbers normalized to `#`).
4. **Quality node.** If `char_count < 80` or `text_density < 0.08`, LangGraph routes that page to OCR. Other pages never pay for OCR.
5. **Structural heuristics.** Font-size delta vs body mode, bold flag, and `1.` / `1.1` / `Chapter` numbering → heading levels → parent stack.
6. **Chunk + index.** Recursive split (~1100 chars, 160 overlap). Each chunk’s `retrieval_text` is `SectionPath\n\nbody`. Vectors go to `vec_baseline`.

`DocumentState` is a typed dict: path, profiles, blocks, sections, chunks, warnings, ocr_pages. Failed OCR is a warning, not a crash.

## 4. What a chunk is

A **window of body text under one heading**, prefixed with the inherited path (`3 Model Architecture > 3.1 Encoder`). Overlap is only the tail of the previous window inside the same section — not the previous section.

Provenance: `page_start`, `page_end`, `section_id`. No bounding boxes on chunks (blocks have them in memory; we do not persist block geometry in this stack).

## 5. Schema this pipeline writes

Shared: `documents`, `pages`, `assets`, `pipeline_runs`, `sections`, `chunks`, `chunks_fts`.

Exclusive index: **`vec_baseline`**.

It does **not** write `document_parameters`, `prism_manifests`, or `chorusgraph_*`.

## 6. Failure modes we accept

| Mode | Behavior |
|---|---|
| Stylish slides, same font everywhere | Headings collapse; chunks become page-ish windows |
| Table-as-image | pdfplumber misses; body text may still contain the numbers |
| Aggressive header strip | A real title repeated on every chapter page can be dropped |
| OCR skew / handwriting | Empty page + `ocr_failed` warning |
| Cross-document questions | No graph; only whatever the single vector space retrieves |

## 7. Tradeoffs

- **LangGraph over ad-hoc if/else** so the OCR branch is a first-class node we can replay.
- **pdfplumber + PyMuPDF, not Unstructured**, to keep the path deterministic and offline.
- **One embedding channel** so bake-off deltas vs Prism’s **six** VectorPrism tables are meaningful.
- **No LLM.** Title comes from PDF metadata or filename. Summaries are empty. Cheap and reproducible.
- **No ChorusGraph.** Cross-doc questions are whatever `vec_baseline` + FTS5 rank. LangGraph here is **per-document** (extract → OCR? → structure → chunk). A second collection graph could run the same cosine bonus Prism uses; we did not add it so Baseline stays the control.

## 8. How the bake-off scores this stack

Same graded rubric as Prism and Relay (not a pass-count):

- Each hit is 0–3 (right PDF + required string + section cues). Distractors (e.g. *The Little Prince* on a voltage query) and neighbor digits (`26 V` without `24 V`) are 0.
- Query score = 40% nDCG@5 + 25% MRR + 15% P@5 + 10% provenance/path + 10% distractor/coverage.
- Overall = 75% retrieval + 25% structure (heading recall vs gold, running-header leak, section-path coverage).

Baseline typically ties Relay on retrieval and loses a little on structure when seed headers (`Confidential draft`, `Studyfetch seed corpus`) leak into `retrieval_text`. It has no 6-channel or graph capability flags — those are Prism-only and **do not add points**.

Gold outlines: Attention paper (Introduction … Conclusion), Nexus-24 (Overview … Safety), Little Prince (Chapter 1 / 5 / 21 / 27).

## 9. Representation (assignment questions)

| Question | Answer | How |
|---|---|---|
| Useful for downstream AI? | Yes — as a control RAG contract | Outline tree + `retrieval_text` with inherited path. One `vec_baseline` channel + FTS5. No signed digits, no graph. A study agent can retrieve a heading-scoped window without re-parsing the PDF. |
| Traceable to the original document? | Yes | `document_id` → filename + sha256. Every chunk has `section_id` and `page_start`/`page_end`. Compare retrieval shows the filename and page range on the hit. |
| Context without unnecessary duplication? | Yes | Prefix is the inherited path only (`3 Model Architecture > 3.1 Encoder`). Recursive windows stay under one heading. Overlap is the tail of the previous window in that section — not sibling sections or the rest of the chapter. |

Compare retrieval attaches this card on every `/api/query` as `design` so the three columns cannot look like the same index.

## 10. With another week

Heading F1 on a larger set; Camelot for ruled tables; persist block bboxes; optional RAPTOR summaries on long sections only; a collection LangGraph if we want Baseline to own the bonus too.
