"""Tests for personiq.extractor — MemoryExtractor with mocked LLM."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from personiq.extractor import MemoryExtractor, _format_convo
from personiq.models import MemoryCategory
from tests.conftest import make_config


@pytest.fixture
def config(tmp_path):
    return make_config(tmp_path)


def _make(response: str, config):
    mock_llm = MagicMock()
    ext = MemoryExtractor(llm=mock_llm, config=config)
    ext._chain = MagicMock()
    ext._chain.invoke  = MagicMock(return_value=response)
    ext._chain.ainvoke = AsyncMock(return_value=response)
    return ext


def _conv(*pairs):
    """Build messages from (user_text, ai_text) pairs. ai_text can be None."""
    out = []
    for u, a in pairs:
        out.append(HumanMessage(content=u))
        if a:
            out.append(AIMessage(content=a))
    return out


# ── format_convo ───────────────────────────────────────────────────────────────

class TestFormatConvo:
    def test_includes_both_roles(self):
        msgs   = _conv(("Hi", "Hello!"), ("I use Python", "Great!"))
        result = _format_convo(msgs, window=4)
        assert "User: Hi"          in result
        assert "Assistant: Hello!" in result
        assert "User: I use Python" in result

    def test_respects_window(self):
        msgs = _conv(("old1", "ok"), ("old2", "ok"), ("recent", "yep"))
        result = _format_convo(msgs, window=1)
        assert "recent" in result
        assert "old1"   not in result

    def test_empty_returns_placeholder(self):
        assert _format_convo([], window=6) == "(no conversation)"


# ── Extraction ─────────────────────────────────────────────────────────────────

class TestExtract:
    def test_valid_facts_parsed(self, config):
        payload = json.dumps({"facts": [
            {"content": "User is a Python developer",
             "category": "technical", "confidence": 0.95, "keywords": ["python"]},
            {"content": "User prefers dark mode",
             "category": "preference", "confidence": 0.8, "keywords": ["dark mode"]},
        ]})
        ext = _make(payload, config)
        r   = ext.extract(_conv(("I write Python and love dark mode", None)))
        assert len(r.facts) == 2
        assert r.facts[0].category == MemoryCategory.TECHNICAL
        assert "python" in r.facts[0].keywords

    def test_empty_facts_list(self, config):
        ext = _make('{"facts": []}', config)
        r   = ext.extract(_conv(("What is the weather?", None)))
        assert r.facts == []

    def test_markdown_fences_stripped(self, config):
        payload = "```json\n" + json.dumps({"facts": [
            {"content": "User likes hiking",
             "category": "preference", "confidence": 0.9, "keywords": ["hiking"]}
        ]}) + "\n```"
        ext = _make(payload, config)
        r   = ext.extract(_conv(("I went hiking", None)))
        assert len(r.facts) == 1

    def test_malformed_json_returns_empty(self, config):
        ext = _make("not json at all", config)
        r   = ext.extract(_conv(("something", None)))
        assert r.facts == []

    def test_category_synonyms_mapped(self, config):
        payload = json.dumps({"facts": [
            {"content": "User is an engineer",
             "category": "background", "confidence": 0.9, "keywords": []},
            {"content": "User likes brevity",
             "category": "communication", "confidence": 0.85, "keywords": []},
        ]})
        ext = _make(payload, config)
        r   = ext.extract(_conv(("I'm an engineer who likes brevity", None)))
        assert r.facts[0].category == MemoryCategory.CONTEXT
        assert r.facts[1].category == MemoryCategory.STYLE

    def test_keywords_as_comma_string(self, config):
        payload = json.dumps({"facts": [
            {"content": "User knows Rust", "category": "technical",
             "confidence": 0.9, "keywords": "rust,systems,programming"}
        ]})
        ext = _make(payload, config)
        r   = ext.extract(_conv(("I write Rust", None)))
        assert "rust" in r.facts[0].keywords

    def test_empty_content_skipped(self, config):
        payload = json.dumps({"facts": [
            {"content": "",           "category": "context",  "confidence": 0.9, "keywords": []},
            {"content": "Valid fact", "category": "goal",     "confidence": 0.8, "keywords": []},
        ]})
        ext = _make(payload, config)
        r   = ext.extract(_conv(("something", None)))
        assert len(r.facts) == 1
        assert r.facts[0].content == "Valid fact"

    def test_empty_messages_returns_empty(self, config):
        ext = _make('{"facts":[]}', config)
        r   = ext.extract([])
        assert r.facts == []

    def test_multi_turn_window(self, config):
        """Extractor should accept a full conversation list."""
        payload = json.dumps({"facts": [
            {"content": "User is a data scientist",
             "category": "technical", "confidence": 0.9, "keywords": ["data"]}
        ]})
        ext  = _make(payload, config)
        msgs = _conv(
            ("I build ML pipelines", "Sounds great!"),
            ("I use Python and SQL", None),
        )
        r = ext.extract(msgs)
        assert len(r.facts) == 1


# ── Async ──────────────────────────────────────────────────────────────────────

class TestAExtract:
    @pytest.mark.asyncio
    async def test_async_works(self, config):
        payload = json.dumps({"facts": [
            {"content": "User works in data science",
             "category": "technical", "confidence": 0.9, "keywords": ["data"]}
        ]})
        ext = _make(payload, config)
        r   = await ext.aextract(_conv(("I build ML pipelines", None)))
        assert len(r.facts) == 1

    @pytest.mark.asyncio
    async def test_async_empty(self, config):
        ext = _make("{}", config)
        r   = await ext.aextract([])
        assert r.facts == []