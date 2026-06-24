"""Tests for personiq.store — CRUD, FTS5, and hybrid search."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from personiq.models import Memory, MemoryCategory
from personiq.store import MemoryStore
from tests.conftest import DIM, make_config


def _vec(seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _mem(user_id="alice", content="User likes Python",
         category=MemoryCategory.TECHNICAL, keywords="python", seed=42, **kw) -> Memory:
    return Memory(user_id=user_id, content=content, category=category,
                  keywords=keywords, embedding=_vec(seed), **kw)


@pytest.fixture
def store(tmp_path: Path):
    cfg = make_config(tmp_path)
    s   = MemoryStore(cfg, embedding_dim=DIM)
    s.connect()
    yield s
    s.close()


# ── CRUD ───────────────────────────────────────────────────────────────────────

class TestCRUD:
    def test_insert_and_get(self, store):
        m = _mem()
        store.upsert(m)
        r = store.get(m.id)
        assert r is not None
        assert r.content  == m.content
        assert r.keywords == m.keywords

    def test_upsert_updates(self, store):
        m = _mem(content="original")
        store.upsert(m)
        m.content    = "updated"
        m.confidence = 0.95
        store.upsert(m)
        assert store.get(m.id).confidence == pytest.approx(0.95)
        assert store.get(m.id).content == "updated"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_count(self, store):
        assert store.count("alice") == 0
        store.upsert(_mem("alice", "F1", seed=1))
        store.upsert(_mem("alice", "F2", seed=2))
        store.upsert(_mem("bob",   "F3", seed=3))
        assert store.count("alice") == 2
        assert store.count("bob")   == 1

    def test_get_all_sorted_by_importance(self, store):
        m1 = _mem("alice", "Low",  seed=1); m1.importance_score = 0.5
        m2 = _mem("alice", "High", seed=2); m2.importance_score = 2.0
        store.upsert(m1); store.upsert(m2)
        rows = store.get_all("alice")
        assert rows[0].content == "High"

    def test_delete(self, store):
        m = _mem()
        store.upsert(m)
        store.delete(m.id)
        assert store.get(m.id) is None

    def test_delete_user(self, store):
        store.upsert(_mem("alice", "A", seed=1))
        store.upsert(_mem("alice", "B", seed=2))
        store.upsert(_mem("bob",   "C", seed=3))
        assert store.delete_user("alice") == 2
        assert store.count("alice") == 0
        assert store.count("bob")   == 1

    def test_get_by_category(self, store):
        store.upsert(_mem("alice", "Go fact",  MemoryCategory.TECHNICAL,  seed=1))
        store.upsert(_mem("alice", "Tea pref", MemoryCategory.PREFERENCE, seed=2))
        tech = store.get_by_category("alice", MemoryCategory.TECHNICAL)
        pref = store.get_by_category("alice", MemoryCategory.PREFERENCE)
        assert len(tech) == 1 and "Go" in tech[0].content
        assert len(pref) == 1

    def test_update_content_syncs_fts(self, store):
        m = _mem(content="User uses Python", keywords="python")
        store.upsert(m)
        store.update_content(m.id, "User uses Python and Go", "python,go")
        assert "Go" in store.get(m.id).content


# ── Decay & access ─────────────────────────────────────────────────────────────

class TestDecay:
    def test_update_after_retrieval(self, store):
        m = _mem(); store.upsert(m)
        m.access_count = 5
        m.decay_factor = 0.8
        store.update_after_retrieval(m.id, m)
        r = store.get(m.id)
        assert r.access_count == 5
        assert r.decay_factor == pytest.approx(0.8)


# ── Hybrid search ──────────────────────────────────────────────────────────────

class TestHybridSearch:
    def test_finds_by_vector(self, store):
        v      = _vec(seed=99)
        target = Memory(user_id="alice", content="User is a Rust developer",
                        category=MemoryCategory.TECHNICAL, keywords="rust,systems",
                        embedding=v)
        store.upsert(target)
        for i in range(4):
            store.upsert(_mem("alice", f"Noise {i}", seed=i + 200))

        results = store.hybrid_search(
            user_id="alice", query_text="rust programming",
            query_embedding=v, top_k=1, threshold=0.0)
        assert len(results) == 1
        assert results[0].memory.id == target.id
        assert results[0].semantic_score > 0.99

    def test_bm25_boosts_keyword_match(self, store):
        v  = _vec(seed=77)
        m1 = Memory(user_id="alice", content="User knows Django",
                    category=MemoryCategory.TECHNICAL,
                    keywords="django,python,web", embedding=v)
        m2 = Memory(user_id="alice", content="User knows Flask",
                    category=MemoryCategory.TECHNICAL,
                    keywords="", embedding=v)
        store.upsert(m1); store.upsert(m2)
        results = store.hybrid_search(
            user_id="alice", query_text="django framework",
            query_embedding=v, top_k=2, threshold=0.0)
        assert results[0].memory.id == m1.id

    def test_user_isolation(self, store):
        v = _vec(seed=5)
        store.upsert(Memory(user_id="alice", content="Alice mem",
                            category=MemoryCategory.CONTEXT, keywords="", embedding=v))
        store.upsert(Memory(user_id="bob",   content="Bob mem",
                            category=MemoryCategory.CONTEXT, keywords="", embedding=v))
        results = store.hybrid_search(
            user_id="alice", query_text="test",
            query_embedding=v, top_k=10, threshold=0.0)
        assert all(r.memory.user_id == "alice" for r in results)

    def test_empty_returns_empty(self, store):
        assert store.hybrid_search(
            user_id="nobody", query_text="anything",
            query_embedding=_vec(), top_k=5, threshold=0.0) == []

    def test_result_has_all_score_fields(self, store):
        m = _mem("alice", "Some fact", seed=10)
        store.upsert(m)
        results = store.hybrid_search(
            user_id="alice", query_text="some fact",
            query_embedding=m.embedding, top_k=1, threshold=0.0)
        assert len(results) == 1
        r = results[0]
        assert 0.0 <= r.recency_score  <= 1.0
        assert r.category_weight >= 1.0
        assert r.final_score     > 0.0