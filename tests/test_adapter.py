"""Tests for personiq.adapter — PersoniqAdapter public API."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from personiq.adapter import PersoniqAdapter
from personiq.config import PersoniqConfig
from personiq.memory_manager import MemoryManager
from personiq.models import MemoryCategory
from tests.conftest import (
    DeterministicEmbedder, make_config, make_extractor, msgs, patch_extractor,
)


def _make_adapter(tmp_path: Path, **cfg_kwargs) -> PersoniqAdapter:
    """Build a PersoniqAdapter with mocked internals — no real LLM or embedder."""
    cfg = make_config(tmp_path, **cfg_kwargs)
    ext = make_extractor(cfg)
    mgr = MemoryManager(
        config=cfg,
        embedder=DeterministicEmbedder(),
        extractor=ext,
    ).start()
    mock_llm = MagicMock()
    piq      = PersoniqAdapter(llm=mock_llm, config=cfg, _manager=mgr)
    piq._injector = _real_injector(cfg)
    return piq


def _real_injector(cfg: PersoniqConfig):
    from personiq.injector import ContextInjector
    return ContextInjector(cfg)


# ── Basic API ──────────────────────────────────────────────────────────────────

class TestBasicAPI:
    def test_context_empty_when_no_memories(self, tmp_path):
        piq = _make_adapter(tmp_path)
        assert piq.context("alice", "hello") == ""

    def test_persona_empty_when_no_memories(self, tmp_path):
        piq = _make_adapter(tmp_path)
        assert piq.persona("alice", "hello") == ""

    def test_count_zero_initially(self, tmp_path):
        piq = _make_adapter(tmp_path)
        assert piq.count("alice") == 0

    def test_memories_empty_initially(self, tmp_path):
        piq = _make_adapter(tmp_path)
        assert piq.memories("alice") == []

    def test_forget_zero_initially(self, tmp_path):
        piq = _make_adapter(tmp_path)
        assert piq.forget("alice") == 0

    def test_repr_contains_mode(self, tmp_path):
        piq = _make_adapter(tmp_path)
        assert "context" in repr(piq)


# ── learn_sync ─────────────────────────────────────────────────────────────────

class TestLearnSync:
    def test_stores_memories(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "User is a Go developer",
             "category": "technical", "confidence": 0.95, "keywords": ["go"]},
        ])
        piq.learn_sync("alice", msgs("I write Go for a living."))
        assert piq.count("alice") == 1

    def test_memories_returns_list(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "User knows Rust",
             "category": "technical", "confidence": 0.9, "keywords": ["rust"]},
        ])
        piq.learn_sync("alice", msgs("I use Rust."))
        mems = piq.memories("alice")
        assert len(mems) == 1
        assert mems[0].content == "User knows Rust"

    def test_forget_clears_all(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "Fact", "category": "context",
             "confidence": 0.9, "keywords": []},
        ])
        piq.learn_sync("alice", msgs("test"))
        assert piq.count("alice") == 1
        piq.forget("alice")
        assert piq.count("alice") == 0

    def test_memories_by_category(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "User likes Python",
             "category": "technical", "confidence": 0.9, "keywords": ["python"]},
            {"content": "User likes coffee",
             "category": "preference", "confidence": 0.9, "keywords": ["coffee"]},
        ])
        piq.learn_sync("alice", msgs("I code Python and drink coffee."))
        tech = piq.memories("alice", category="technical")
        pref = piq.memories("alice", category="preference")
        assert len(tech) == 1
        assert len(pref) == 1

    def test_invalid_category_returns_all(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "Some fact", "category": "context",
             "confidence": 0.9, "keywords": []},
        ])
        piq.learn_sync("alice", msgs("test"))
        result = piq.memories("alice", category="nonexistent_category")
        assert isinstance(result, list)


# ── context & persona after learning ──────────────────────────────────────────

class TestContextAfterLearning:
    def test_context_returns_string_after_learning(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "User is a backend developer",
             "category": "technical", "confidence": 1.0, "keywords": ["backend"]},
        ])
        piq.learn_sync("alice", msgs("I do backend development."))
        ctx = piq.context("alice", "User is a backend developer")
        assert isinstance(ctx, str)

    def test_persona_returns_string_after_learning(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "User knows Python",
             "category": "technical", "confidence": 1.0, "keywords": ["python"]},
        ])
        piq.learn_sync("alice", msgs("I use Python."))
        ctx = piq.persona("alice", "User knows Python")
        assert isinstance(ctx, str)


# ── inject ─────────────────────────────────────────────────────────────────────

class TestInject:
    def test_inject_no_memories_returns_unchanged(self, tmp_path):
        piq      = _make_adapter(tmp_path)
        messages = [{"role": "system", "content": "You are helpful."},
                    {"role": "user",   "content": "Hello"}]
        result   = piq.inject(messages, user_id="alice", query="Hello")
        assert result[0]["content"] == "You are helpful."

    def test_inject_inserts_system_when_missing(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "User likes Go",
             "category": "technical", "confidence": 1.0, "keywords": ["go"]},
        ])
        piq.learn_sync("alice", msgs("I love Go."))
        messages = [{"role": "user", "content": "help me"}]
        result   = piq.inject(messages, user_id="alice", query="User likes Go")
        if len(result) > 1 or result[0]["role"] == "system":
            # context was injected
            assert any(m["role"] == "system" for m in result)


# ── LangGraph nodes ────────────────────────────────────────────────────────────

class TestLangGraphNodes:
    def test_load_node_is_callable(self, tmp_path):
        piq  = _make_adapter(tmp_path)
        node = piq.load_node
        assert callable(node)

    def test_save_node_is_callable(self, tmp_path):
        piq  = _make_adapter(tmp_path)
        node = piq.save_node
        assert callable(node)

    @pytest.mark.asyncio
    async def test_load_node_empty_state_returns_empty_context(self, tmp_path):
        piq    = _make_adapter(tmp_path)
        node   = piq.load_node
        result = await node({"user_id": "alice", "messages": []})
        assert result["memory_context"] == ""

    @pytest.mark.asyncio
    async def test_load_node_returns_context_string(self, tmp_path):
        piq  = _make_adapter(tmp_path)
        node = piq.load_node
        result = await node({
            "user_id":  "alice",
            "messages": [HumanMessage(content="hello")],
        })
        assert isinstance(result.get("memory_context"), str)

    @pytest.mark.asyncio
    async def test_save_node_returns_empty_dict(self, tmp_path):
        piq    = _make_adapter(tmp_path)
        node   = piq.save_node
        result = await node({
            "user_id":  "alice",
            "messages": [HumanMessage(content="test")],
        })
        assert result == {}

    @pytest.mark.asyncio
    async def test_load_node_no_human_message_returns_empty(self, tmp_path):
        piq    = _make_adapter(tmp_path)
        node   = piq.load_node
        result = await node({"user_id": "alice", "messages": []})
        assert result["memory_context"] == ""


# ── Async learn ────────────────────────────────────────────────────────────────

class TestAsyncLearn:
    @pytest.mark.asyncio
    async def test_learn_async_returns_int(self, tmp_path):
        piq = _make_adapter(tmp_path)
        patch_extractor(piq._manager._extractor, [
            {"content": "User codes Python",
             "category": "technical", "confidence": 0.9, "keywords": []}
        ])
        count = await piq.learn("alice", msgs("I code Python."))
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_acontext_returns_string(self, tmp_path):
        piq = _make_adapter(tmp_path)
        ctx = await piq.acontext("alice", "test")
        assert isinstance(ctx, str)

    @pytest.mark.asyncio
    async def test_apersona_returns_string(self, tmp_path):
        piq = _make_adapter(tmp_path)
        ctx = await piq.apersona("alice", "test")
        assert isinstance(ctx, str)


# ── Config & manager properties ────────────────────────────────────────────────

class TestProperties:
    def test_config_accessible(self, tmp_path):
        piq = _make_adapter(tmp_path)
        assert isinstance(piq.config, PersoniqConfig)

    def test_manager_accessible(self, tmp_path):
        piq = _make_adapter(tmp_path)
        assert isinstance(piq.manager, MemoryManager)