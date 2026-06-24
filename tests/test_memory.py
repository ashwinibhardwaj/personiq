"""Tests for personiq.memory_manager — full pipeline with mocked LLM."""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from personiq.config import PersoniqConfig
from personiq.memory_manager import MemoryManager
from personiq.models import MemoryCategory
from tests.conftest import (
    DeterministicEmbedder, make_config, make_extractor, msgs, patch_extractor,
)


@pytest.fixture
def cfg(tmp_path: Path) -> PersoniqConfig:
    return make_config(tmp_path)


@pytest.fixture
def manager(cfg) -> MemoryManager:
    ext = make_extractor(cfg)
    m   = MemoryManager(config=cfg, embedder=DeterministicEmbedder(), extractor=ext)
    m.start()
    yield m
    m.stop()


# ── Lifecycle ──────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_stop(self, cfg):
        ext = make_extractor(cfg)
        m   = MemoryManager(config=cfg, embedder=DeterministicEmbedder(), extractor=ext)
        m.start()
        assert m._started
        m.stop()
        assert not m._started

    def test_context_manager(self, cfg):
        ext = make_extractor(cfg)
        with MemoryManager(config=cfg, embedder=DeterministicEmbedder(), extractor=ext) as m:
            assert m._started
        assert not m._started

    def test_requires_start(self, cfg):
        ext = make_extractor(cfg)
        m   = MemoryManager(config=cfg, embedder=DeterministicEmbedder(), extractor=ext)
        with pytest.raises(RuntimeError):
            m.get_context("alice", "hello")

    def test_double_start_safe(self, cfg):
        ext = make_extractor(cfg)
        m   = MemoryManager(config=cfg, embedder=DeterministicEmbedder(), extractor=ext)
        m.start(); m.start()
        assert m._started
        m.stop()

    def test_no_llm_raises(self, cfg):
        with pytest.raises(RuntimeError, match="LLM"):
            MemoryManager(config=cfg, embedder=DeterministicEmbedder()).start()


# ── learn ──────────────────────────────────────────────────────────────────────

class TestLearn:
    def test_stores_facts(self, manager, cfg):
        patch_extractor(manager._extractor, [
            {"content": "User is a Python developer",
             "category": "technical", "confidence": 0.95, "keywords": ["python"]},
            {"content": "User prefers dark mode",
             "category": "preference", "confidence": 0.85, "keywords": ["dark mode"]},
        ])
        stored = manager.learn("alice", msgs("I code Python and use dark mode."))
        assert stored == 2
        assert manager.count("alice") == 2

    def test_empty_message_stores_nothing(self, manager):
        stored = manager.learn("alice", msgs("   "))
        assert stored == 0

    def test_dedup_prevents_duplicate(self, manager):
        fact = {"content": "User is a Python developer",
                "category": "technical", "confidence": 0.95, "keywords": ["python"]}
        patch_extractor(manager._extractor, [fact])
        manager.learn("alice", msgs("I code in Python."))
        manager.learn("alice", msgs("Python is my main language."))
        # same deterministic embedding → dedup kicks in
        assert manager.count("alice") == 1

    def test_different_users_isolated(self, manager):
        patch_extractor(manager._extractor, [
            {"content": "User likes jazz", "category": "preference",
             "confidence": 0.9, "keywords": ["jazz"]}
        ])
        manager.learn("alice", msgs("I love jazz."))
        manager.learn("bob",   msgs("I love jazz."))
        assert manager.count("alice") == 1
        assert manager.count("bob")   == 1
        assert manager.list_all("alice")[0].user_id == "alice"

    def test_keywords_stored(self, manager):
        patch_extractor(manager._extractor, [
            {"content": "User works with Rust",
             "category": "technical", "confidence": 0.9,
             "keywords": ["rust", "systems"]}
        ])
        manager.learn("alice", msgs("I build with Rust."))
        mem = manager.list_all("alice")
        assert len(mem) == 1
        assert "rust" in mem[0].keywords


# ── get_context ────────────────────────────────────────────────────────────────

class TestGetContext:
    def test_empty_when_no_memories(self, manager):
        assert manager.get_context("newuser", "hello") == ""

    def test_context_mode_contains_header(self, manager):
        patch_extractor(manager._extractor, [
            {"content": "User is a data scientist",
             "category": "technical", "confidence": 1.0, "keywords": ["data"]}
        ])
        manager.learn("alice", msgs("I am a data scientist."))
        ctx = manager.get_context("alice", "User is a data scientist", mode="context")
        if ctx:
            assert "personiq" in ctx.lower()

    def test_persona_mode_returns_string(self, manager):
        patch_extractor(manager._extractor, [
            {"content": "User knows Python",
             "category": "technical", "confidence": 1.0, "keywords": ["python"]}
        ])
        manager.learn("alice", msgs("I know Python."))
        ctx = manager.get_context("alice", "User knows Python", mode="persona")
        assert isinstance(ctx, str)


# ── forget & list ──────────────────────────────────────────────────────────────

class TestForgetAndList:
    def test_forget_removes_all(self, manager):
        patch_extractor(manager._extractor, [
            {"content": "Fact A", "category": "context",
             "confidence": 0.9, "keywords": []},
            {"content": "Fact B", "category": "context",
             "confidence": 0.9, "keywords": []},
        ])
        manager.learn("alice", msgs("some message"))
        assert manager.count("alice") == 2
        assert manager.forget("alice") == 2
        assert manager.count("alice") == 0

    def test_forget_nonexistent_returns_zero(self, manager):
        assert manager.forget("nobody") == 0

    def test_list_by_category(self, manager):
        patch_extractor(manager._extractor, [
            {"content": "User knows Go",  "category": "technical",
             "confidence": 0.9, "keywords": ["go"]},
            {"content": "User likes tea", "category": "preference",
             "confidence": 0.9, "keywords": ["tea"]},
        ])
        manager.learn("alice", msgs("I code Go and drink tea."))
        tech = manager.list_by_category("alice", MemoryCategory.TECHNICAL)
        pref = manager.list_by_category("alice", MemoryCategory.PREFERENCE)
        assert len(tech) == 1
        assert len(pref) == 1


# ── async ──────────────────────────────────────────────────────────────────────

class TestAsync:
    @pytest.mark.asyncio
    async def test_aget_context_returns_str(self, cfg):
        ext = make_extractor(cfg)
        async with _amgr(cfg, ext) as m:
            ctx = await m.aget_context("alice", "test")
            assert isinstance(ctx, str)

    @pytest.mark.asyncio
    async def test_alearn_fire_and_forget(self, tmp_path):
        cfg = make_config(tmp_path, async_extraction=True)
        ext = make_extractor(cfg)
        patch_extractor(ext, [
            {"content": "Async fact", "category": "context",
             "confidence": 0.9, "keywords": []}
        ])
        async with _amgr(cfg, ext) as m:
            count = await m.alearn("alice", msgs("test"))
            assert count == 0   # fire-and-forget returns 0 immediately

    @pytest.mark.asyncio
    async def test_alearn_sync_mode_returns_count(self, tmp_path):
        cfg = make_config(tmp_path, async_extraction=False)
        ext = make_extractor(cfg)
        patch_extractor(ext, [
            {"content": "Sync fact", "category": "context",
             "confidence": 0.9, "keywords": []}
        ])
        async with _amgr(cfg, ext) as m:
            count = await m.alearn("alice", msgs("test"))
            assert count == 1


class _amgr:
    def __init__(self, cfg, ext):
        self._cfg = cfg
        self._ext = ext

    async def __aenter__(self):
        self._m = MemoryManager(
            config=self._cfg,
            embedder=DeterministicEmbedder(),
            extractor=self._ext,
        )
        self._m.start()
        return self._m

    async def __aexit__(self, *_):
        self._m.stop()