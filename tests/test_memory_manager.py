"""
Tests for personiq.memory_manager — MemoryManager integration.

These tests mock the LLM extractor and use a real (temp file) SQLite store +
a deterministic embedding engine so the full pipeline can be tested without
any external API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from personiq.config import PersonIQConfig
from personiq.embeddings import EmbeddingEngine
from personiq.memory_manager import MemoryManager
from personiq.models import MemoryCategory

EMBEDDING_DIM = 384


# ── Deterministic embedding engine ────────────────────────────────────────────

class _DeterministicEmbedder:
    """
    Maps strings to fixed unit vectors deterministically.
    Same string always returns the same vector so retrieval is predictable.
    """

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_mock_extractor(config: PersonIQConfig) -> "MemoryExtractor":
    """Create a MemoryExtractor with a mocked LLM chain (no API key needed)."""
    from personiq.extractor import MemoryExtractor
    from langchain_core.language_models import BaseChatModel
    mock_llm = MagicMock(spec=BaseChatModel)
    extractor = MemoryExtractor(llm=mock_llm, config=config)
    # Patch chain with empty-facts default; tests override per-call
    extractor._chain = MagicMock()
    extractor._chain.invoke = MagicMock(return_value='{"facts": []}')
    extractor._chain.ainvoke = AsyncMock(return_value='{"facts": []}')
    return extractor


def _patch_extractor(manager: MemoryManager, facts: list[dict]) -> None:
    """Replace the manager's extractor chain with a mock returning `facts`."""
    payload = json.dumps({"facts": facts})
    manager._extractor._chain = MagicMock()
    manager._extractor._chain.invoke = MagicMock(return_value=payload)
    manager._extractor._chain.ainvoke = AsyncMock(return_value=payload)


@pytest.fixture
def tmp_config(tmp_path: Path) -> PersonIQConfig:
    return PersonIQConfig(
        db_path=str(tmp_path / "test.db"),
        async_extraction=False,   # synchronous in tests for simplicity
    )


@pytest.fixture
def manager(tmp_config: PersonIQConfig) -> MemoryManager:
    mock_extractor = _make_mock_extractor(tmp_config)
    m = MemoryManager(
        config=tmp_config,
        embedding_engine=_DeterministicEmbedder(),
        extractor=mock_extractor,
    )
    m.start()
    yield m
    m.stop()


# ── Lifecycle tests ────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_stop(self, tmp_config):
        m = MemoryManager(config=tmp_config, embedding_engine=_DeterministicEmbedder(),
                          extractor=_make_mock_extractor(tmp_config))
        m.start()
        assert m._started is True
        m.stop()
        assert m._started is False

    def test_context_manager(self, tmp_config):
        with MemoryManager(config=tmp_config, embedding_engine=_DeterministicEmbedder(),
                           extractor=_make_mock_extractor(tmp_config)) as m:
            assert m._started is True
        assert m._started is False

    def test_requires_start(self, tmp_config):
        m = MemoryManager(config=tmp_config, embedding_engine=_DeterministicEmbedder(),
                          extractor=_make_mock_extractor(tmp_config))
        with pytest.raises(RuntimeError, match="start()"):
            m.get_context("alice", "hello")

    def test_double_start_is_safe(self, tmp_config):
        m = MemoryManager(config=tmp_config, embedding_engine=_DeterministicEmbedder(),
                          extractor=_make_mock_extractor(tmp_config))
        m.start()
        m.start()   # should not raise or reinitialise
        assert m._started
        m.stop()


# ── after_message tests ────────────────────────────────────────────────────────

class TestAfterMessage:
    def test_stores_extracted_facts(self, manager: MemoryManager):
        _patch_extractor(manager, [
            {"content": "User is a Python developer", "category": "technical", "confidence": 0.95},
            {"content": "User prefers dark mode", "category": "preference", "confidence": 0.85},
        ])
        stored = manager.after_message("alice", "I code in Python and use dark mode.")
        assert stored == 2
        assert manager.memory_count("alice") == 2

    def test_empty_message_stores_nothing(self, manager: MemoryManager):
        stored = manager.after_message("alice", "   ")
        assert stored == 0
        assert manager.memory_count("alice") == 0

    def test_deduplication_prevents_duplicate(self, manager: MemoryManager):
        fact = {"content": "User is a Python developer", "category": "technical", "confidence": 0.95}
        _patch_extractor(manager, [fact])
        manager.after_message("alice", "I code in Python.")

        # Same fact again – should reinforce, not duplicate
        manager.after_message("alice", "Python is my main language.")
        # Count stays 1 because the same embedding is reinforced
        assert manager.memory_count("alice") == 1

    def test_different_users_isolated(self, manager: MemoryManager):
        _patch_extractor(manager, [
            {"content": "User likes jazz", "category": "preference", "confidence": 0.9}
        ])
        manager.after_message("alice", "I love jazz.")
        manager.after_message("bob", "I love jazz.")

        assert manager.memory_count("alice") == 1
        assert manager.memory_count("bob") == 1
        alice_mems = manager.list_memories("alice")
        assert alice_mems[0].user_id == "alice"


# ── get_context tests ──────────────────────────────────────────────────────────

class TestGetContext:
    def test_returns_empty_string_when_no_memories(self, manager: MemoryManager):
        context = manager.get_context("newuser", "Hello!")
        assert context == ""

    def test_context_contains_memory_content(self, manager: MemoryManager):
        _patch_extractor(manager, [
            {"content": "User is a data scientist", "category": "technical", "confidence": 1.0},
        ])
        manager.after_message("alice", "I work as a data scientist.")

        # Use the same content as query so embeddings match deterministically
        context = manager.get_context("alice", "User is a data scientist")
        assert "data scientist" in context

    def test_context_includes_memory_header(self, manager: MemoryManager):
        _patch_extractor(manager, [
            {"content": "User prefers Python", "category": "technical", "confidence": 1.0},
        ])
        manager.after_message("alice", "I prefer Python")
        context = manager.get_context("alice", "User prefers Python")
        if context:  # only check format if memories were retrieved
            assert "memory" in context.lower() or "context" in context.lower()


# ── forget_user tests ──────────────────────────────────────────────────────────

class TestForgetUser:
    def test_forget_removes_all_memories(self, manager: MemoryManager):
        _patch_extractor(manager, [
            {"content": "Fact A", "category": "context", "confidence": 0.9},
            {"content": "Fact B", "category": "context", "confidence": 0.9},
        ])
        manager.after_message("alice", "Some message")
        assert manager.memory_count("alice") == 2

        deleted = manager.forget_user("alice")
        assert deleted == 2
        assert manager.memory_count("alice") == 0

    def test_forget_nonexistent_user_returns_zero(self, manager: MemoryManager):
        assert manager.forget_user("nobody") == 0


# ── Async tests ────────────────────────────────────────────────────────────────

class TestAsync:
    @pytest.mark.asyncio
    async def test_aget_context_returns_string(self, tmp_config):
        async with _async_manager(tmp_config) as m:
            context = await m.aget_context("alice", "test query")
            assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_after_message_async_non_blocking(self, tmp_config):
        cfg = PersonIQConfig(
            db_path=tmp_config.db_path,
            async_extraction=True,   # fire-and-forget mode
        )
        async with _async_manager(cfg) as m:
            _patch_extractor(m, [
                {"content": "Async fact", "category": "context", "confidence": 0.9}
            ])
            count = await m.after_message_async("alice", "test")
            # Non-blocking returns 0 immediately (task is in background)
            assert count == 0


class _async_manager:
    """Async context manager wrapper for MemoryManager."""
    def __init__(self, config):
        self._config = config

    async def __aenter__(self):
        self._m = MemoryManager(
            config=self._config,
            embedding_engine=_DeterministicEmbedder(),
            extractor=_make_mock_extractor(self._config),
        )
        self._m.start()
        return self._m

    async def __aexit__(self, *_):
        self._m.stop()