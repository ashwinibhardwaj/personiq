"""Shared test fixtures."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from personiq.config import PersoniqConfig
from personiq.extractor import MemoryExtractor

DIM = 384


class DeterministicEmbedder:
    """Same string always → same unit vector. No model download needed."""
    @property
    def dimension(self) -> int: return DIM

    def embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 31))
        v   = rng.standard_normal(DIM).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def make_config(tmp_path: Path, **kwargs) -> PersoniqConfig:
    defaults = dict(async_extraction=False, retrieval_queries=1)
    defaults.update(kwargs)
    return PersoniqConfig(db_path=str(tmp_path / "test.db"), **defaults)


def make_extractor(config: PersoniqConfig) -> MemoryExtractor:
    mock_llm = MagicMock()
    ext = MemoryExtractor(llm=mock_llm, config=config)
    ext._chain = MagicMock()
    ext._chain.invoke  = MagicMock(return_value='{"facts": []}')
    ext._chain.ainvoke = AsyncMock(return_value='{"facts": []}')
    return ext


def patch_extractor(ext: MemoryExtractor, facts: list[dict]) -> None:
    payload = json.dumps({"facts": facts})
    ext._chain.invoke  = MagicMock(return_value=payload)
    ext._chain.ainvoke = AsyncMock(return_value=payload)


def msgs(*texts: str):
    return [HumanMessage(content=t) for t in texts]