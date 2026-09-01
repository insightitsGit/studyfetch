# Master Design Document 2: The Prism Stack

## 1. Executive summary

Prism is an **enterprise GraphRAG + zero-trust** stack for heterogeneous collections (papers, textbooks, financial/technical sheets). It replaces a single flat index with:

- **PrismCortex** — runtime intent routing (academic / textbook / financial / technical)
- **ChorusGraph** — intra-doc hierarchy plus **cross-document** entity and section edges
- **VectorPrism** — six embedding channels (semantic, structural, title, entity, numeric, caption) plus FTS5
- **PrismManifest + PrismShield** — Ed25519-signed parameters; Shield flags *drifted* metrics only

Binaries (source PDFs, figures) go to **Azure Blob** when configured, otherwise the same path scheme on disk (`{document_id}/p{page}_{xref}.png`). All relational and vector state lives in shared `intelligence.db`.

## 2. Technical stack

| Component | Implementation |
|---|---|
| PrismCortex | Layout + lexicon classifier, then column-aware vs parameter-first parse |
| Assets | PyMuPDF images → Blob; tables as markdown + JSON matrix |
| ChorusGraph | `chorusgraph_nodes` / `chorusgraph_edges`; `same_entity`, `overlaps_with`, `contains`, `subsumes`, `mentions`, `defines` |
| VectorPrism | six `vec_prism_*` tables (`float[384]` each): semantic, structural, title, entity, numeric, caption |
| PrismManifest | Ed25519 (PKCS8 PEM under `DATA_DIR/prism_ed25519.pem`) |
| PrismShield | Applied on `/api/query` for `pipeline_id=prism` |
| Lexical | Shared `chunks_fts` |

## 3. Processing pipeline

1. **PrismCortex.** Sample text + page profiles. `financial` if revenue/EBITDA/$; `technical` if voltage/datasheet; `academic` if abstract/DOI; `textbook` if chapter/exercise. Academic/textbook → column reorder + section tree. Financial/technical → same geometry plus aggressive parameter harvest.
2. **Asset separation.** Figures uploaded to blob store; tables stored as headers/rows/markdown on `assets`. Not inlined into every chunk.
3. **ChorusGraph.** Document / section / entity / parameter nodes. Entity ids are **per document** (`node_ent_{doc}_{hash(label)}`) so the same label in two PDFs is two nodes that can be linked. Cross-doc: exact label links (`same_entity`) and cosine ≥ **0.70** on section title+summary (`overlaps_with`). The seed pair that justifies 0.70 (not 0.78) is Firmware Notes ↔ 3.1 Encoder (≈0.769). This is the assignment bonus: related sections with different wording still connect; leftover Prism sections are **unique**.
4. **PrismManifest.** Regex/name-value harvest (`Maximum Operating Voltage: 24 V`, `Q4 Revenue: $1,042,500`). Each row is canonical-JSON hashed and Ed25519-signed. A document-level manifest signs the list of param hashes.
5. **VectorPrism.** Each chunk is embedded into six subspaces (same 384-d model, different text):
   - **semantic** — full `retrieval_text`
   - **structural** — section path
   - **title** — document + section title
   - **entity** — extracted entities in the chunk
   - **numeric** — bound metrics (`Maximum Operating Voltage=24 V`)
   - **caption** — figure/table captions on those pages
   At query time every channel is kNN-searched. Intent-weighted reciprocal-rank fusion mixes them (parameter/financial → numeric; academic/outline → title+structural; default → semantic). FTS and ChorusGraph expand after the mix.
6. **PrismShield (Ask / `/api/query` only).** This is **not** an ingest step. After VectorPrism mix and ChorusGraph expand, each number in a hit is classified against the signed manifest: **verified** (exact signed value), **unsigned** (years, F1, section numbers — left alone), or **drifted** (same bound unit as a signed param, different value that was never bound — e.g. invented `23 V` next to signed `24 V`). Retrieval text is not rewritten by default; optional `rewrite_drift` marks drifted tokens as `[DRIFT:…]`. Table min/typ/max sharing a unit are signed so real neighbors like `26 V` are not treated as drift. The query payload includes `shield` (rolled-up verdict) plus per-hit `shield` and `verified_parameters`. Baseline and Relay never run this gate.

## 4. Storage

**Blob:** `az://document-assets/{document_id}/…` or `file://{DATA_DIR}/blobs/…`.

**SQLite (exclusive to Prism, plus shared chunks):**

```sql
document_parameters (param_id, document_id, section_id, parameter_name,
                     numeric_value, raw_string_value, unit, data_type,
                     provenance_page, manifest_id, manifest_signature)
prism_manifests (manifest_id, document_id, payload_hash, signature, public_key)
chorusgraph_nodes / chorusgraph_edges
vec_prism_semantic / structural / title / entity / numeric / caption
```

## 5. Chunking and provenance

Chunks are **section-scoped recursive splits** (~900 chars, light overlap) with prefix:

`Document: {title} / Intent: {academic|…} / Section: {path}`.

Every chunk keeps `page_start`/`page_end`/`section_id`. Every parameter keeps `provenance_page` + signature. Graph edges keep both document ids.

## 6. Retrieval contract (downstream)

A Prism hit may include:

- `retrieval_text` (source text; Shield does not blank signed or prose numbers)
- `channels` used (semantic / structural / title / entity / numeric / caption / fts / chorusgraph)
- `vectorprism.weights` for the intent that was applied
- `graph_edge` / `graph_weight` if expanded
- `verified_parameters[]` with raw string, float, page
- `shield.verified_parameters[]` / `shield.unsigned[]` / `shield.drifted[]`

Downstream agents should **prefer `verified_parameters` over numbers in prose**.

## 7. Failure modes

| Mode | Defense / residual risk |
|---|---|
| Digit hallucination | Signed `document_parameters`; Shield flags same-unit values that were never bound |
| Over-Shield | Mitigated: unsigned prose (F1, years, `3.1`) is left intact; only bound-unit drift is marked |
| Entity NER | Capitalization heuristic, not spaCy; misses lowercase concepts |
| Cross-doc false edges | 0.70 cosine can still link generic “Introduction” sections |
| Intent misroute | Datasheet without keywords classified academic; still extracts parameters |
| Key loss | If `prism_ed25519.pem` is rotated, old signatures fail verification |

## 8. Bonus surface (cross-document)

After each Prism run, `_link_cross_document` compares this PDF’s sections to every other Prism section and writes edges. `GET /api/bonus` (Benchmark tab) rolls those up:

| Output | Meaning |
|---|---|
| Related sections | `overlaps_with` / `same_entity` with a **why** (cosine on title+summary, or shared label) |
| Document pairs | Section-level edges rolled up (`edge_count`, max/mean weight) |
| Unique sections | Prism sections with no cross-doc overlap edge |

At query time, retrieve kNN-searches all six channels, fuses by intent, then expands the top hits along `overlaps_with` / `same_entity` and tags `chorusgraph` even if the neighbor was already in the list.

On the current library: 3 overlap edges (Firmware Notes ↔ Encoder / paper title / Multi-Head). `same_entity` is empty — the seeds share *meaning*, not capitalized labels. Entity matching is a capitalization heuristic, not NER.

## 9. How the bake-off scores this stack

Same graded 0–3 / nDCG / MRR / P@5 rubric as Baseline and Relay. **Capability is listed, not added to the 100:** `vectorprism_6ch`, `vectorprism_numeric_mix`, `chorusgraph_related`. `prism_verified_digits` was removed so Shield is not a fake extra 10 points.

Prism usually wins structure (no running-header leak in `retrieval_text`) and parameter queries (numeric channel). It can lose nDCG on the overlap query when graph expand reorders a slightly weaker hit.

## 10. Tradeoffs

- **Six vec0 tables, not a packed tensor.** sqlite-vec has no native multi-channel tensor; VectorPrism is six `vec0` tables plus intent-weighted rank fusion. Honest and benchmarkable.
- **Crypto on parameters only**, not every sentence. Signing prose does not stop hallucinations; signing digits does.
- **Shield flags, it does not rewrite.** Unsigned prose stays. Drift = same bound unit, value never signed.
- **Graph is incremental.** Re-running Prism on a new PDF links it into the existing ChorusGraph. Deleting a doc removes its nodes and edges.
- **Azure is optional.** Same code path; local Docker uses the filesystem so the assignment runs without a subscription.

## 11. Representation (assignment questions)

| Question | Answer | How |
|---|---|---|
| Useful for downstream AI? | Yes — richer than a flat RAG hit | Six VectorPrism channels + intent mix, `verified_parameters` (prefer these over prose digits), `shield` (verified / unsigned / drifted), and ChorusGraph edges (`graph_edge`, `graph_weight`, `channels` includes `chorusgraph`). |
| Traceable to the original document? | Yes, including digits | Pages + `section_id` on every chunk. Each signed metric has `provenance_page` + Ed25519. Graph edges keep both document ids. Filename is attached at query time. |
| Context without unnecessary duplication? | Yes | Prefix is `Document / Intent / Section` only. Figures stay as `asset_ids`, not pixels in every chunk. Graph expand adds **one related section** from another PDF — not the sibling chapter and not the whole collection. |

Compare retrieval shows the live mix weights, channels, Shield, and any `overlaps_with` expand so this column is visibly not Baseline.

## 12. With another week

LayoutLMv3 page type; proper NER; per-parameter type checkers (currency vs volts); graph drawing; a scanned seed so OCR is visible; Shield allowlist for citations/years.
