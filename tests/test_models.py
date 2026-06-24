"""Tests for personiq.models — Memory lifecycle and scoring."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from personiq.models import (
    ExtractedFact, ExtractionResult, HybridSearchResult,
    Memory, MemoryCategory, CATEGORY_WEIGHT,
)


class TestMemory:
    def test_default_fields(self):
        m = Memory(user_id="alice", content="User likes Python",
                   category=MemoryCategory.TECHNICAL)
        assert m.confidence      == 1.0
        assert m.access_count    == 0
        assert m.decay_factor    == 1.0
        assert m.importance_score == 1.0
        assert m.id is not None

    def test_content_stripped(self):
        m = Memory(user_id="alice", content="  trimmed  ",
                   category=MemoryCategory.CONTEXT)
        assert m.content == "trimmed"

    def test_empty_content_raises(self):
        with pytest.raises(Exception):
            Memory(user_id="alice", content="   ", category=MemoryCategory.CONTEXT)

    def test_touch_increments_access(self):
        m = Memory(user_id="alice", content="fact", category=MemoryCategory.CONTEXT)
        m.touch()
        assert m.access_count == 1
        assert m.confidence   >= 1.0   # capped at 1.0 when starting at 1.0

    def test_touch_bumps_confidence(self):
        m = Memory(user_id="alice", content="fact",
                   category=MemoryCategory.CONTEXT, confidence=0.7)
        m.touch(confidence_bump=0.1)
        assert m.confidence == pytest.approx(0.8)

    def test_touch_caps_confidence_at_one(self):
        m = Memory(user_id="alice", content="fact",
                   category=MemoryCategory.CONTEXT, confidence=0.99)
        m.touch(confidence_bump=0.5)
        assert m.confidence == pytest.approx(1.0)

    def test_apply_decay_reduces_factor(self):
        m = Memory(user_id="alice", content="fact", category=MemoryCategory.CONTEXT)
        # Backdate updated_at by 100 days
        m.updated_at = datetime.now(timezone.utc) - timedelta(days=100)
        m.apply_decay(decay_rate=0.005)
        assert m.decay_factor < 1.0
        assert m.decay_factor > 0.0

    def test_fresh_memory_decay_near_one(self):
        m = Memory(user_id="alice", content="fact", category=MemoryCategory.CONTEXT)
        m.apply_decay(decay_rate=0.005)
        assert m.decay_factor > 0.99   # just created → minimal decay

    def test_recompute_importance_uses_all_signals(self):
        m = Memory(user_id="alice", content="fact",
                   category=MemoryCategory.CONTEXT, confidence=0.8)
        m.access_count = 10
        m.decay_factor = 0.9
        m._recompute()
        # importance should reflect all three signals
        assert m.importance_score > 0
        assert m.importance_score < 5   # sanity upper bound

    def test_unique_ids(self):
        m1 = Memory(user_id="a", content="fact1", category=MemoryCategory.CONTEXT)
        m2 = Memory(user_id="a", content="fact2", category=MemoryCategory.CONTEXT)
        assert m1.id != m2.id


class TestCategoryWeights:
    def test_all_categories_have_weight(self):
        for cat in MemoryCategory:
            assert cat in CATEGORY_WEIGHT or cat.value in CATEGORY_WEIGHT

    def test_technical_highest(self):
        assert CATEGORY_WEIGHT[MemoryCategory.TECHNICAL] >= max(
            v for k, v in CATEGORY_WEIGHT.items() if k != MemoryCategory.TECHNICAL
        )

    def test_all_weights_positive(self):
        assert all(v > 0 for v in CATEGORY_WEIGHT.values())


class TestExtractedFact:
    def test_default_confidence(self):
        f = ExtractedFact(content="fact", category=MemoryCategory.GOAL)
        assert f.confidence == pytest.approx(0.9)

    def test_content_stripped(self):
        f = ExtractedFact(content="  stripped  ", category=MemoryCategory.GOAL)
        assert f.content == "stripped"

    def test_keywords_default_empty(self):
        f = ExtractedFact(content="fact", category=MemoryCategory.GOAL)
        assert f.keywords == []


class TestExtractionResult:
    def test_default_empty_facts(self):
        r = ExtractionResult()
        assert r.facts == []

    def test_stores_facts(self):
        f = ExtractedFact(content="fact", category=MemoryCategory.CONTEXT)
        r = ExtractionResult(facts=[f])
        assert len(r.facts) == 1


class TestHybridSearchResult:
    def test_default_scores(self):
        m = Memory(user_id="a", content="fact", category=MemoryCategory.CONTEXT)
        r = HybridSearchResult(memory=m)
        assert r.semantic_score  == 0.0
        assert r.bm25_score      == 0.0
        assert r.recency_score   == 1.0
        assert r.category_weight == 1.0
        assert r.final_score     == 0.0