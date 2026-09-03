from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import sqlite_vec

from app.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_VEC_TABLES = {
    "baseline": "vec_baseline",
    "prism_semantic": "vec_prism_semantic",
    "prism_structural": "vec_prism_structural",
    "prism_title": "vec_prism_title",
    "prism_entity": "vec_prism_entity",
    "prism_numeric": "vec_prism_numeric",
    "prism_caption": "vec_prism_caption",
    "relay": "vec_relay",
}

_lock = threading.Lock()
_store: "Store | None" = None


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.conn = _connect(path)
        self._init_schema()

    def _init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(sql)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(benchmark_queries)").fetchall()}
        if "gold_json" not in cols:
            self.conn.execute(
                "ALTER TABLE benchmark_queries ADD COLUMN gold_json TEXT NOT NULL DEFAULT '{}'"
            )
        dim = settings.embedding_dim
        for table in _VEC_TABLES.values():
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
                f"chunk_id TEXT PRIMARY KEY, embedding FLOAT[{dim}])"
            )
        self.conn.commit()

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq: list) -> sqlite3.Cursor:
        return self.conn.executemany(sql, seq)

    def fetchall(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def fetchone(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def upsert_document(self, doc: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO documents (id, filename, sha256, title, page_count, metadata_json, blob_uri, created_at)
            VALUES (:id, :filename, :sha256, :title, :page_count, :metadata_json, :blob_uri, :created_at)
            ON CONFLICT(sha256) DO UPDATE SET
                filename=excluded.filename,
                title=excluded.title,
                page_count=excluded.page_count,
                metadata_json=excluded.metadata_json,
                blob_uri=excluded.blob_uri
            """,
            doc,
        )
        self.conn.commit()

    def replace_pages(self, document_id: str, pages: list[dict[str, Any]]) -> None:
        self.conn.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
        self.conn.executemany(
            """
            INSERT INTO pages (id, document_id, page_number, char_count, image_coverage,
                               text_density, label, features_json, text_preview)
            VALUES (:id, :document_id, :page_number, :char_count, :image_coverage,
                    :text_density, :label, :features_json, :text_preview)
            """,
            pages,
        )
        self.conn.commit()

    def replace_assets(self, document_id: str, assets: list[dict[str, Any]]) -> None:
        self.conn.execute("DELETE FROM assets WHERE document_id = ?", (document_id,))
        if assets:
            self.conn.executemany(
                """
                INSERT INTO assets (id, document_id, page_number, asset_type, caption,
                                    blob_uri, bbox_json, extra_json)
                VALUES (:id, :document_id, :page_number, :asset_type, :caption,
                        :blob_uri, :bbox_json, :extra_json)
                """,
                assets,
            )
        self.conn.commit()

    def create_run(self, run: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO pipeline_runs (id, pipeline_id, document_id, status, started_at,
                                       finished_at, stats_json, warnings_json, error)
            VALUES (:id, :pipeline_id, :document_id, :status, :started_at,
                    :finished_at, :stats_json, :warnings_json, :error)
            """,
            run,
        )
        self.conn.commit()

    def finish_run(
        self,
        run_id: str,
        status: str,
        stats: dict,
        warnings: list,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE pipeline_runs
            SET status=?, finished_at=?, stats_json=?, warnings_json=?, error=?
            WHERE id=?
            """,
            (status, utcnow(), json.dumps(stats), json.dumps(warnings), error, run_id),
        )
        self.conn.commit()

    def delete_run_outputs(self, document_id: str, pipeline_id: str) -> None:
        prior = self.fetchall(
            "SELECT id FROM pipeline_runs WHERE document_id=? AND pipeline_id=?",
            (document_id, pipeline_id),
        )
        run_ids = [r["id"] for r in prior]
        if not run_ids:
            return
        placeholders = ",".join("?" * len(run_ids))
        chunk_ids = [
            r["id"]
            for r in self.fetchall(
                f"SELECT id FROM chunks WHERE run_id IN ({placeholders})",
                tuple(run_ids),
            )
        ]
        self.conn.execute(
            f"DELETE FROM sections WHERE run_id IN ({placeholders})", tuple(run_ids)
        )
        self.conn.execute(
            f"DELETE FROM chunks WHERE run_id IN ({placeholders})", tuple(run_ids)
        )
        if pipeline_id == "prism":
            self.conn.execute(
                "DELETE FROM document_parameters WHERE document_id=? AND pipeline_id=?",
                (document_id, pipeline_id),
            )
            self.conn.execute(
                "DELETE FROM prism_manifests WHERE document_id=?", (document_id,)
            )
            self.conn.execute(
                "DELETE FROM chorusgraph_nodes WHERE document_id=?", (document_id,)
            )
            self.conn.execute(
                """
                DELETE FROM chorusgraph_edges
                WHERE document_id_source=? OR document_id_target=?
                """,
                (document_id, document_id),
            )
        if chunk_ids:
            cph = ",".join("?" * len(chunk_ids))
            self.conn.execute(
                f"DELETE FROM embeddings_meta WHERE chunk_id IN ({cph})",
                tuple(chunk_ids),
            )
            for table in self._vec_tables_for(pipeline_id):
                self.conn.execute(
                    f"DELETE FROM {table} WHERE chunk_id IN ({cph})",
                    tuple(chunk_ids),
                )
            self.conn.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({cph})",
                tuple(chunk_ids),
            )
        self.conn.execute(
            f"DELETE FROM pipeline_runs WHERE id IN ({placeholders})", tuple(run_ids)
        )
        self.conn.commit()

    def delete_document(self, document_id: str) -> bool:
        doc = self.fetchone("SELECT * FROM documents WHERE id=?", (document_id,))
        if not doc:
            return False
        from app.storage.blobs import blob_store
        from app.config import settings

        for pipeline_id in ("baseline", "prism", "relay"):
            self.delete_run_outputs(document_id, pipeline_id)
        leftover = [r["id"] for r in self.fetchall("SELECT id FROM chunks WHERE document_id=?", (document_id,))]
        if leftover:
            cph = ",".join("?" * len(leftover))
            self.conn.execute(f"DELETE FROM embeddings_meta WHERE chunk_id IN ({cph})", tuple(leftover))
            self.conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({cph})", tuple(leftover))
            for table in _VEC_TABLES.values():
                try:
                    self.conn.execute(f"DELETE FROM {table} WHERE chunk_id IN ({cph})", tuple(leftover))
                except Exception:
                    pass
        self.conn.execute("DELETE FROM sections WHERE document_id=?", (document_id,))
        self.conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        self.conn.execute("DELETE FROM pipeline_runs WHERE document_id=?", (document_id,))
        self.conn.execute("DELETE FROM pages WHERE document_id=?", (document_id,))
        self.conn.execute("DELETE FROM assets WHERE document_id=?", (document_id,))
        self.conn.execute("DELETE FROM document_parameters WHERE document_id=?", (document_id,))
        self.conn.execute("DELETE FROM prism_manifests WHERE document_id=?", (document_id,))
        self.conn.execute("DELETE FROM chorusgraph_nodes WHERE document_id=?", (document_id,))
        self.conn.execute(
            "DELETE FROM chorusgraph_edges WHERE document_id_source=? OR document_id_target=?",
            (document_id, document_id),
        )
        self.conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        self.conn.commit()
        blob_store.delete_prefix(document_id)
        extra = settings.upload_dir / f"{document_id}.pdf"
        if extra.exists():
            extra.unlink()
        return True

    def _vec_tables_for(self, pipeline_id: str) -> list[str]:
        if pipeline_id == "prism":
            return [
                "vec_prism_semantic",
                "vec_prism_structural",
                "vec_prism_title",
                "vec_prism_entity",
                "vec_prism_numeric",
                "vec_prism_caption",
            ]
        if pipeline_id == "baseline":
            return ["vec_baseline"]
        if pipeline_id == "relay":
            return ["vec_relay"]
        return []

    def insert_sections(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO sections (id, run_id, document_id, pipeline_id, parent_id, level,
                                  title, page_start, page_end, text, summary, extra_json)
            VALUES (:id, :run_id, :document_id, :pipeline_id, :parent_id, :level,
                    :title, :page_start, :page_end, :text, :summary, :extra_json)
            """,
            rows,
        )
        self.conn.commit()

    def insert_chunks(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO chunks (id, run_id, document_id, pipeline_id, section_id, chunk_index,
                                text, retrieval_text, page_start, page_end, token_estimate,
                                context_json, asset_ids_json)
            VALUES (:id, :run_id, :document_id, :pipeline_id, :section_id, :chunk_index,
                    :text, :retrieval_text, :page_start, :page_end, :token_estimate,
                    :context_json, :asset_ids_json)
            """,
            rows,
        )
        self.conn.executemany(
            """
            INSERT INTO chunks_fts (chunk_id, pipeline_id, document_id, retrieval_text)
            VALUES (?, ?, ?, ?)
            """,
            [
                (r["id"], r["pipeline_id"], r["document_id"], r["retrieval_text"])
                for r in rows
            ],
        )
        self.conn.commit()

    def insert_vectors(
        self, table: str, items: list[tuple[str, list[float]]], pipeline_id: str, channel: str
    ) -> None:
        if table not in _VEC_TABLES.values():
            raise ValueError(f"unknown vec table {table}")
        if not items:
            return
        self.conn.executemany(
            f"INSERT INTO {table} (chunk_id, embedding) VALUES (?, ?)",
            [(cid, sqlite_vec.serialize_float32(vec)) for cid, vec in items],
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings_meta (chunk_id, pipeline_id, channel, dim)
            VALUES (?, ?, ?, ?)
            """,
            [(cid, pipeline_id, channel, len(vec)) for cid, vec in items],
        )
        self.conn.commit()

    def knn(
        self,
        table: str,
        query_vec: list[float],
        k: int = 8,
        pipeline_id: str | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if table not in _VEC_TABLES.values():
            raise ValueError(f"unknown vec table {table}")
        blob = sqlite_vec.serialize_float32(query_vec)
        try:
            fetch_k = k * 8 if document_id else k
            neighbors = self.fetchall(
                f"SELECT chunk_id, distance FROM {table} WHERE embedding MATCH ? AND k = ?",
                (blob, fetch_k),
            )
        except sqlite3.Error:
            fetch_k = k * 8 if document_id else k
            neighbors = self.fetchall(
                f"""
                SELECT chunk_id, distance FROM {table}
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (blob, fetch_k),
            )
        out = []
        for n in neighbors:
            chunk = self.fetchone("SELECT * FROM chunks WHERE id=?", (n["chunk_id"],))
            if not chunk:
                continue
            if pipeline_id and chunk["pipeline_id"] != pipeline_id:
                continue
            if document_id and chunk["document_id"] != document_id:
                continue
            out.append(
                {
                    **chunk,
                    "chunk_id": n["chunk_id"],
                    "distance": n["distance"],
                }
            )
            if len(out) >= k:
                break
        return out

    def fts(
        self, query: str, pipeline_id: str, k: int = 8, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        import re

        tokens = [t for t in re.findall(r"[A-Za-z0-9]{2,}", query) if t.lower() not in {"the", "and", "for", "what", "how"}]
        if not tokens:
            return []
        match = " OR ".join(tokens)
        try:
            return self.fetchall(
                """
                SELECT c.id AS chunk_id, c.retrieval_text, c.document_id, c.pipeline_id,
                       c.page_start, c.page_end, c.section_id, c.context_json, c.asset_ids_json,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ? AND c.pipeline_id = ?
                  AND (? IS NULL OR c.document_id = ?)
                ORDER BY score
                LIMIT ?
                """,
                (match, pipeline_id, document_id, document_id, k),
            )
        except sqlite3.Error:
            return []

    def document_structure(self, document_id: str, pipeline_id: str) -> dict[str, Any] | None:
        doc = self.fetchone("SELECT * FROM documents WHERE id=?", (document_id,))
        if not doc:
            return None
        run = self.fetchone(
            """
            SELECT * FROM pipeline_runs
            WHERE document_id=? AND pipeline_id=?
            ORDER BY started_at DESC LIMIT 1
            """,
            (document_id, pipeline_id),
        )
        sections = self.fetchall(
            """
            SELECT * FROM sections
            WHERE document_id=? AND pipeline_id=?
            ORDER BY page_start, level
            """,
            (document_id, pipeline_id),
        )
        chunks = self.fetchall(
            """
            SELECT * FROM chunks
            WHERE document_id=? AND pipeline_id=?
            ORDER BY page_start, chunk_index
            """,
            (document_id, pipeline_id),
        )
        pages = self.fetchall(
            "SELECT * FROM pages WHERE document_id=? ORDER BY page_number",
            (document_id,),
        )
        assets = self.fetchall(
            "SELECT * FROM assets WHERE document_id=? ORDER BY page_number",
            (document_id,),
        )
        params = self.fetchall(
            "SELECT * FROM document_parameters WHERE document_id=? ORDER BY provenance_page",
            (document_id,),
        )
        nodes = self.fetchall(
            "SELECT * FROM chorusgraph_nodes WHERE document_id=?",
            (document_id,),
        )
        edges = self.fetchall(
            """
            SELECT * FROM chorusgraph_edges
            WHERE document_id_source=? OR document_id_target=?
            """,
            (document_id, document_id),
        )
        return {
            "document": doc,
            "run": run,
            "pages": pages,
            "sections": sections,
            "chunks": chunks,
            "assets": assets,
            "parameters": params,
            "graph_nodes": nodes,
            "graph_edges": edges,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        docs = self.fetchall("SELECT * FROM documents ORDER BY created_at DESC")
        for doc in docs:
            runs = self.fetchall(
                """
                SELECT pipeline_id, status, finished_at, stats_json
                FROM pipeline_runs WHERE document_id=?
                ORDER BY started_at DESC
                """,
                (doc["id"],),
            )
            seen: set[str] = set()
            latest = []
            for r in runs:
                if r["pipeline_id"] in seen:
                    continue
                seen.add(r["pipeline_id"])
                latest.append(r)
            doc["runs"] = latest
            raw = doc.get("metadata_json") or "{}"
            try:
                doc["metadata"] = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                doc["metadata"] = {}
        return docs

    def index_stats(self) -> list[dict[str, Any]]:
        stats = []
        for name, table in _VEC_TABLES.items():
            row = self.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            stats.append({"index": name, "table": table, "vectors": row["n"] if row else 0})
        fts = self.fetchone("SELECT COUNT(*) AS n FROM chunks_fts")
        stats.append({"index": "fts5", "table": "chunks_fts", "vectors": fts["n"] if fts else 0})
        return stats

    def close(self) -> None:
        self.conn.close()


def get_store() -> Store:
    global _store
    with _lock:
        if _store is None:
            settings.ensure_dirs()
            _store = Store(settings.db_path)
        return _store
