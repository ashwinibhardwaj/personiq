"""
personiq.store
~~~~~~~~~~~~~~
SQLite storage with FTS5 (BM25) + vector (cosine) hybrid search
fused via Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from personiq.config import PersoniqConfig
from personiq.models import CATEGORY_WEIGHT, HybridSearchResult, Memory, MemoryCategory

logger  = logging.getLogger(__name__)
_RRF_K  = 60
_STOPS  = {"the","a","an","is","in","on","at","to","for","of","and","or","but",
           "it","this","that","with","as","be","was","are","i","my","me","we"}


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id               = row["id"],
        user_id          = row["user_id"],
        content          = row["content"],
        category         = MemoryCategory(row["category"]),
        confidence       = row["confidence"],
        access_count     = row["access_count"],
        decay_factor     = row["decay_factor"],
        importance_score = row["importance_score"],
        keywords         = row["keywords"] or "",
        source_text      = row["source_text"],
        created_at       = datetime.fromisoformat(row["created_at"]),
        updated_at       = datetime.fromisoformat(row["updated_at"]),
    )


class MemoryStore:
    def __init__(self, config: PersoniqConfig, embedding_dim: int) -> None:
        self._config        = config
        self._dim           = embedding_dim
        self._conn: Optional[sqlite3.Connection] = None
        self._vec_available = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        Path(self._config.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._config.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA cache_size=-32000")
        self._conn.execute("PRAGMA temp_store=MEMORY")

        try:
            import sqlite_vec
            sqlite_vec.load(self._conn)
            self._vec_available = True
            logger.info("[personiq] sqlite-vec loaded: %s", sqlite_vec.__file__)
        except Exception as e:
            logger.warning("[personiq] sqlite-vec unavailable (%s) — brute-force cosine active", e)

        self._migrate()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def _db(self) -> sqlite3.Connection:
        if not self._conn:
            raise RuntimeError("MemoryStore: call connect() first")
        return self._conn

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        self._db.executescript(f"""
            CREATE TABLE IF NOT EXISTS memories (
                id               TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                content          TEXT NOT NULL,
                category         TEXT NOT NULL,
                confidence       REAL    NOT NULL DEFAULT 1.0,
                access_count     INTEGER NOT NULL DEFAULT 0,
                decay_factor     REAL    NOT NULL DEFAULT 1.0,
                importance_score REAL    NOT NULL DEFAULT 1.0,
                keywords         TEXT    NOT NULL DEFAULT '',
                source_text      TEXT,
                created_at       TEXT    NOT NULL,
                updated_at       TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_m_user     ON memories (user_id);
            CREATE INDEX IF NOT EXISTS idx_m_user_cat ON memories (user_id, category);
            CREATE INDEX IF NOT EXISTS idx_m_user_imp ON memories (user_id, importance_score DESC);

            CREATE TABLE IF NOT EXISTS memory_vectors (
                id        TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(id UNINDEXED, user_id UNINDEXED, content, keywords);
        """)
        if self._vec_available:
            self._db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories
                USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[{self._dim}])
            """)
        self._db.commit()


    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert(self, m: Memory) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cat = m.category if isinstance(m.category, str) else m.category.value
        self._db.execute("""
            INSERT INTO memories
              (id,user_id,content,category,confidence,access_count,
               decay_factor,importance_score,keywords,source_text,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              content=excluded.content, confidence=excluded.confidence,
              access_count=excluded.access_count, decay_factor=excluded.decay_factor,
              importance_score=excluded.importance_score, keywords=excluded.keywords,
              updated_at=excluded.updated_at
        """, (m.id, m.user_id, m.content, cat, m.confidence, m.access_count,
              m.decay_factor, m.importance_score, m.keywords, m.source_text,
              m.created_at.isoformat(), now))

        try:
            self._db.execute("DELETE FROM memory_fts WHERE id=?", (m.id,))
            self._db.execute(
                "INSERT INTO memory_fts(id,user_id,content,keywords) VALUES(?,?,?,?)",
                (m.id, m.user_id, m.content, m.keywords))
        except Exception as e:
            logger.debug("[personiq] FTS sync warning: %s", e)

        if m.embedding:
            blob = np.array(m.embedding, dtype=np.float32).tobytes()
            self._db.execute("""
                INSERT INTO memory_vectors(id,embedding) VALUES(?,?)
                ON CONFLICT(id) DO UPDATE SET embedding=excluded.embedding
            """, (m.id, blob))
            if self._vec_available:
                self._db.execute("""
                    INSERT INTO vec_memories(id,embedding) VALUES(?,?)
                    ON CONFLICT(id) DO UPDATE SET embedding=excluded.embedding
                """, (m.id, json.dumps(m.embedding)))

        self._db.commit()

    def update_after_retrieval(self, mid: str, m: Memory) -> None:
        self._db.execute("""
            UPDATE memories SET access_count=?,decay_factor=?,importance_score=?,updated_at=?
            WHERE id=?
        """, (m.access_count, m.decay_factor, m.importance_score,
              datetime.now(timezone.utc).isoformat(), mid))
        self._db.commit()

    def update_confidence(self, mid: str, m: Memory) -> None:
        self._db.execute("""
            UPDATE memories SET confidence=?,access_count=?,importance_score=?,updated_at=?
            WHERE id=?
        """, (m.confidence, m.access_count, m.importance_score,
              datetime.now(timezone.utc).isoformat(), mid))
        self._db.commit()

    def update_content(self, mid: str, content: str, keywords: str) -> None:
        self._db.execute(
            "UPDATE memories SET content=?,keywords=?,updated_at=? WHERE id=?",
            (content, keywords, datetime.now(timezone.utc).isoformat(), mid))
        self._db.execute("DELETE FROM memory_fts WHERE id=?", (mid,))
        row = self._db.execute("SELECT user_id FROM memories WHERE id=?", (mid,)).fetchone()
        if row:
            self._db.execute(
                "INSERT INTO memory_fts(id,user_id,content,keywords) VALUES(?,?,?,?)",
                (mid, row["user_id"], content, keywords))
        self._db.commit()

    def delete(self, mid: str) -> None:
        self._db.execute("DELETE FROM memory_fts WHERE id=?", (mid,))
        self._db.execute("DELETE FROM memories WHERE id=?", (mid,))
        self._db.commit()

    def delete_user(self, user_id: str) -> int:
        self._db.execute("DELETE FROM memory_fts WHERE user_id=?", (user_id,))
        cur = self._db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        self._db.commit()
        return cur.rowcount

    # ── Read ───────────────────────────────────────────────────────────────────

    def get(self, mid: str) -> Optional[Memory]:
        row = self._db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        return _row_to_memory(row) if row else None

    def get_all(self, user_id: str) -> list[Memory]:
        rows = self._db.execute(
            "SELECT * FROM memories WHERE user_id=? ORDER BY importance_score DESC",
            (user_id,)).fetchall()
        return [_row_to_memory(r) for r in rows]

    def get_by_category(self, user_id: str, category: MemoryCategory) -> list[Memory]:
        cat  = category.value if hasattr(category, "value") else category
        rows = self._db.execute(
            "SELECT * FROM memories WHERE user_id=? AND category=? ORDER BY importance_score DESC",
            (user_id, cat)).fetchall()
        return [_row_to_memory(r) for r in rows]

    def count(self, user_id: str) -> int:
        return self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id=?", (user_id,)).fetchone()[0]

    def get_vectors(self, user_id: str) -> list[tuple[str, np.ndarray]]:
        rows = self._db.execute("""
            SELECT m.id, mv.embedding FROM memories m
            JOIN memory_vectors mv ON mv.id=m.id WHERE m.user_id=?
        """, (user_id,)).fetchall()
        return [(r["id"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows]

    # ── Hybrid search ──────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        user_id:         str,
        query_text:      str,
        query_embedding: list[float],
        top_k:           int   = 5,
        threshold:       float = 0.20,
        semantic_weight: float = 0.60,
        bm25_weight:     float = 0.30,
        recency_weight:  float = 0.10,
    ) -> list[HybridSearchResult]:
        """
        BM25 (FTS5) + cosine (vector) hybrid search fused with RRF.
        Scores are boosted by memory recency and category weight.
        """
        qvec = np.array(query_embedding, dtype=np.float32)

        # ── BM25 via FTS5 ──────────────────────────────────────────────────────
        bm25_rank: dict[str, int]   = {}
        bm25_raw:  dict[str, float] = {}
        try:
            fts = self._db.execute("""
                SELECT id, rank FROM memory_fts
                WHERE user_id=? AND memory_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, (user_id, _fts_query(query_text), top_k * 4)).fetchall()
            if fts:
                raws    = [abs(float(r["rank"])) for r in fts]
                max_raw = max(raws) or 1.0
                for i, r in enumerate(fts):
                    bm25_rank[r["id"]] = i
                    bm25_raw[r["id"]]  = abs(float(r["rank"])) / max_raw
        except Exception as e:
            logger.debug("[personiq] BM25 error (non-fatal): %s", e)

        # ── Cosine / ANN ───────────────────────────────────────────────────────
        sem_rank:  dict[str, int]   = {}
        sem_score: dict[str, float] = {}
        vec_rows = (
            self._ann_search(user_id, qvec, top_k * 4)
            if self._vec_available
            else self._brute_cosine(user_id, qvec, top_k * 4)
        )
        for i, (mid, sim) in enumerate(vec_rows):
            sem_rank[mid]  = i
            sem_score[mid] = sim

        # ── RRF fusion ─────────────────────────────────────────────────────────
        all_ids = set(bm25_rank) | set(sem_rank)
        if not all_ids:
            return []

        rrf: dict[str, float] = {}
        n = len(all_ids)
        for mid in all_ids:
            sr = sem_rank.get(mid, n + _RRF_K)
            br = bm25_rank.get(mid, n + _RRF_K)
            rrf[mid] = (
                semantic_weight * (1.0 / (_RRF_K + sr)) +
                bm25_weight     * (1.0 / (_RRF_K + br))
            )

        # ── Load, decay, score ─────────────────────────────────────────────────
        ph   = ",".join("?" * len(all_ids))
        rows = self._db.execute(
            f"SELECT * FROM memories WHERE id IN ({ph})", list(all_ids)).fetchall()

        out: list[HybridSearchResult] = []
        for row in rows:
            mem = _row_to_memory(row)
            sem = sem_score.get(mem.id, 0.0)
            if sem < threshold and mem.id not in bm25_rank:
                continue

            mem.apply_decay(self._config.decay_rate)
            mem.access_count += 1
            mem._recompute()

            cw    = CATEGORY_WEIGHT.get(mem.category, 1.0)
            final = rrf.get(mem.id, 0.0) * (1.0 + recency_weight * mem.decay_factor) * cw

            out.append(HybridSearchResult(
                memory          = mem,
                semantic_score  = sem,
                bm25_score      = bm25_raw.get(mem.id, 0.0),
                recency_score   = mem.decay_factor,
                category_weight = cw,
                final_score     = final,
            ))
            self.update_after_retrieval(mem.id, mem)

        out.sort(key=lambda r: r.final_score, reverse=True)
        return out[:top_k]

    # ── Vector internals ───────────────────────────────────────────────────────

    def _ann_search(self, user_id: str, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        try:
            rows = self._db.execute("""
                SELECT v.id, v.distance FROM vec_memories v
                JOIN memories m ON m.id=v.id WHERE m.user_id=?
                AND v.embedding MATCH ? AND k=? ORDER BY v.distance
            """, (user_id, json.dumps(qvec.tolist()), k)).fetchall()
            return [(r["id"], min(1.0, 1.0 - float(r["distance"]))) for r in rows]
        except Exception as e:
            logger.debug("[personiq] ANN fallback: %s", e)
            return self._brute_cosine(user_id, qvec, k)

    def _brute_cosine(self, user_id: str, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        rows = self._db.execute("""
            SELECT m.id, mv.embedding FROM memories m
            JOIN memory_vectors mv ON mv.id=m.id WHERE m.user_id=?
        """, (user_id,)).fetchall()

        scored = []

        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)

            # embeddings already normalized
            sim = float(np.dot(qvec, v))

            scored.append((r["id"], sim))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]


def _fts_query(text: str) -> str:
    tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
    tokens = list(dict.fromkeys(t for t in tokens if t not in _STOPS))
    return " OR ".join(tokens) if tokens else '""'