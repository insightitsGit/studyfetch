-- Shared intelligence.db — all three pipelines write here.
-- Indexes are namespaced by pipeline_id so we can benchmark them side-by-side.

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    sha256          TEXT NOT NULL UNIQUE,
    title           TEXT,
    page_count      INTEGER NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    blob_uri        TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id),
    page_number     INTEGER NOT NULL,
    char_count      INTEGER NOT NULL DEFAULT 0,
    image_coverage  REAL NOT NULL DEFAULT 0,
    text_density    REAL NOT NULL DEFAULT 0,
    label           TEXT,
    features_json   TEXT NOT NULL DEFAULT '{}',
    text_preview    TEXT,
    UNIQUE(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS assets (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id),
    page_number     INTEGER NOT NULL,
    asset_type      TEXT NOT NULL,
    caption         TEXT,
    blob_uri        TEXT,
    bbox_json       TEXT,
    extra_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              TEXT PRIMARY KEY,
    pipeline_id     TEXT NOT NULL,
    document_id     TEXT NOT NULL REFERENCES documents(id),
    status          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    stats_json      TEXT NOT NULL DEFAULT '{}',
    warnings_json   TEXT NOT NULL DEFAULT '[]',
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_doc_pipeline
    ON pipeline_runs(document_id, pipeline_id);

CREATE TABLE IF NOT EXISTS sections (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES pipeline_runs(id),
    document_id     TEXT NOT NULL,
    pipeline_id     TEXT NOT NULL,
    parent_id       TEXT,
    level           INTEGER NOT NULL DEFAULT 1,
    title           TEXT NOT NULL,
    page_start      INTEGER NOT NULL,
    page_end        INTEGER NOT NULL,
    text            TEXT,
    summary         TEXT,
    extra_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sections_run ON sections(run_id);

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES pipeline_runs(id),
    document_id     TEXT NOT NULL,
    pipeline_id     TEXT NOT NULL,
    section_id      TEXT,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    retrieval_text  TEXT NOT NULL,
    page_start      INTEGER NOT NULL,
    page_end        INTEGER NOT NULL,
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    context_json    TEXT NOT NULL DEFAULT '{}',
    asset_ids_json  TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_chunks_pipeline ON chunks(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_chunks_run ON chunks(run_id);

CREATE TABLE IF NOT EXISTS embeddings_meta (
    chunk_id        TEXT NOT NULL,
    pipeline_id     TEXT NOT NULL,
    channel         TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, channel)
);

-- Prism-specific: signed numerical / financial / part-number parameters
CREATE TABLE IF NOT EXISTS document_parameters (
    param_id            TEXT PRIMARY KEY,
    document_id         TEXT NOT NULL,
    section_id          TEXT,
    pipeline_id         TEXT NOT NULL DEFAULT 'prism',
    parameter_name      TEXT NOT NULL,
    numeric_value       REAL,
    raw_string_value    TEXT NOT NULL,
    unit                TEXT,
    data_type           TEXT,
    provenance_page     INTEGER,
    manifest_id         TEXT,
    manifest_signature  TEXT
);

CREATE INDEX IF NOT EXISTS idx_params_doc ON document_parameters(document_id);

CREATE TABLE IF NOT EXISTS prism_manifests (
    manifest_id     TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    payload_hash    TEXT NOT NULL,
    signature       TEXT NOT NULL,
    public_key      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- ChorusGraph nodes + edges (cross-document)
CREATE TABLE IF NOT EXISTS chorusgraph_nodes (
    node_id         TEXT PRIMARY KEY,
    node_type       TEXT NOT NULL,
    document_id     TEXT,
    label           TEXT NOT NULL,
    extra_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chorusgraph_edges (
    edge_id             TEXT PRIMARY KEY,
    source_node         TEXT NOT NULL,
    target_node         TEXT NOT NULL,
    relationship_type   TEXT NOT NULL,
    document_id_source  TEXT,
    document_id_target  TEXT,
    weight              REAL NOT NULL DEFAULT 0,
    extra_json          TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON chorusgraph_edges(source_node);
CREATE INDEX IF NOT EXISTS idx_edges_target ON chorusgraph_edges(target_node);

-- Benchmark harness
CREATE TABLE IF NOT EXISTS benchmark_queries (
    id              TEXT PRIMARY KEY,
    query           TEXT NOT NULL,
    intent          TEXT,
    notes           TEXT,
    gold_json       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS benchmark_results (
    id              TEXT PRIMARY KEY,
    query_id        TEXT NOT NULL,
    pipeline_id     TEXT NOT NULL,
    index_name      TEXT NOT NULL,
    latency_ms      REAL NOT NULL,
    hit_count       INTEGER NOT NULL,
    result_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    pipeline_id,
    document_id,
    retrieval_text
);
