"""Tests for personiq.injector — context and persona output formatting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from personiq.injector import ContextInjector
from personiq.models import HybridSearchResult, Memory, MemoryCategory
from tests.conftest import DIM, make_config


def _hsresult(content: str, category: MemoryCategory,
              semantic: float = 0.8, final: float = 0.8) -> HybridSearchResult:
    rng = np.random.default_rng(abs(hash(content)) % (2 ** 31))
    v   = rng.standard_normal(DIM).astype(np.float32)
    v  /= np.linalg.norm(v)
    mem = Memory(
        user_id="alice", content=content, category=category,
        confidence=0.9, embedding=v.tolist(),
    )
    return HybridSearchResult(
        memory=mem,
        semantic_score=semantic,
        bm25_score=0.3,
        recency_score=0.9,
        category_weight=1.0,
        final_score=final,
    )


@pytest.fixture
def injector(tmp_path: Path) -> ContextInjector:
    return ContextInjector(make_config(tmp_path))


# ── build_context ──────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_empty_results_returns_empty_string(self, injector):
        assert injector.build_context([]) == ""

    def test_contains_header_and_footer(self, injector):
        results = [_hsresult("User knows Python", MemoryCategory.TECHNICAL)]
        ctx     = injector.build_context(results)
        assert "[personiq:" in ctx
        assert "[end of personiq context]" in ctx

    def test_contains_memory_content(self, injector):
        results = [_hsresult("User knows Python", MemoryCategory.TECHNICAL)]
        ctx     = injector.build_context(results)
        assert "User knows Python" in ctx

    def test_groups_by_category(self, injector):
        results = [
            _hsresult("User knows Go",    MemoryCategory.TECHNICAL),
            _hsresult("User likes music", MemoryCategory.PREFERENCE),
        ]
        ctx = injector.build_context(results)
        assert "Technical" in ctx
        assert "Preference" in ctx

    def test_low_confidence_shows_hint(self, injector):
        rng = np.random.default_rng(1)
        v   = rng.standard_normal(DIM).astype(np.float32); v /= np.linalg.norm(v)
        mem = Memory(user_id="alice", content="Uncertain fact",
                     category=MemoryCategory.CONTEXT, confidence=0.5, embedding=v.tolist())
        r   = HybridSearchResult(memory=mem, semantic_score=0.7, bm25_score=0.2,
                                  recency_score=0.9, category_weight=1.0, final_score=0.6)
        ctx = injector.build_context([r])
        assert "confidence" in ctx

    def test_respects_max_context_chars(self, tmp_path):
        cfg = make_config(tmp_path, max_context_chars=100)
        inj = ContextInjector(cfg)
        results = [
            _hsresult("A" * 50 + " fact", MemoryCategory.TECHNICAL),
            _hsresult("B" * 50 + " fact", MemoryCategory.GOAL),
            _hsresult("C" * 50 + " fact", MemoryCategory.PREFERENCE),
        ]
        ctx = inj.build_context(results)
        assert len(ctx) <= 200   # truncation adds footer, so allow some headroom


# ── build_persona ──────────────────────────────────────────────────────────────

class TestBuildPersona:
    def test_empty_results_returns_empty(self, injector):
        assert injector.build_persona([]) == ""

    def test_contains_you_already_know(self, injector):
        results = [_hsresult("User is a Python developer", MemoryCategory.TECHNICAL)]
        ctx     = injector.build_persona(results)
        assert "You already know" in ctx

    def test_natural_language_not_bullets(self, injector):
        results = [_hsresult("User works with FastAPI", MemoryCategory.TECHNICAL)]
        ctx     = injector.build_persona(results)
        # persona mode should NOT produce bullet characters
        assert "•" not in ctx

    def test_contains_memory_topic(self, injector):
        results = [_hsresult("User works with FastAPI", MemoryCategory.TECHNICAL)]
        ctx     = injector.build_persona(results)
        # Should mention the technology somehow
        assert "FastAPI" in ctx or "fastapi" in ctx.lower()

    def test_multi_category_combined(self, injector):
        results = [
            _hsresult("User uses Python",    MemoryCategory.TECHNICAL),
            _hsresult("User wants to launch a SaaS", MemoryCategory.GOAL),
        ]
        ctx = injector.build_persona(results)
        assert isinstance(ctx, str) and len(ctx) > 10


# ── inject_into_messages ───────────────────────────────────────────────────────

class TestInjectIntoMessages:
    def test_no_results_unchanged(self, injector):
        msgs = [{"role": "system", "content": "You are helpful."}]
        out  = injector.inject_into_messages(msgs, [])
        assert out[0]["content"] == "You are helpful."

    def test_prepends_to_existing_system(self, injector):
        msgs    = [{"role": "system", "content": "Base prompt."}]
        results = [_hsresult("User knows Python", MemoryCategory.TECHNICAL)]
        out     = injector.inject_into_messages(msgs, results, mode="context")
        assert "Base prompt." in out[0]["content"]
        assert "personiq"     in out[0]["content"].lower()

    def test_inserts_system_when_none_exists(self, injector):
        msgs    = [{"role": "user", "content": "Hello"}]
        results = [_hsresult("User knows Python", MemoryCategory.TECHNICAL)]
        out     = injector.inject_into_messages(msgs, results, mode="context")
        assert out[0]["role"] == "system"

    def test_persona_mode_natural_language(self, injector):
        msgs    = [{"role": "system", "content": "Base."}]
        results = [_hsresult("User works with Go", MemoryCategory.TECHNICAL)]
        out     = injector.inject_into_messages(msgs, results, mode="persona")
        # persona mode should contain natural language indicator
        assert "You already know" in out[0]["content"]

    def test_does_not_mutate_original(self, injector):
        original = [{"role": "system", "content": "original"}]
        results  = [_hsresult("User knows Python", MemoryCategory.TECHNICAL)]
        injector.inject_into_messages(original, results)
        assert original[0]["content"] == "original"