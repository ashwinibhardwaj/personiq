"""
personiq.embeddings
~~~~~~~~~~~~~~~~~~~
Embedding engine abstraction — local (sentence-transformers) or OpenAI.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from personiq.config import PersoniqConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingEngine(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model      = None
        self._dim: int | None = None

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("pip install sentence-transformers") from e
            logger.info("[personiq] loading embedding model '%s'", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = self._model.get_embedding_dimension()
            else:
                self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        self._load(); return self._dim  # type: ignore

    def embed(self, text: str) -> list[float]:
        self._load()
        return self._model.encode(text, normalize_embeddings=True).tolist()  # type: ignore

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts: return []
        self._load()
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()  # type: ignore


class OpenAIEmbeddingEngine:
    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        self._model_name = model_name
        self._client     = None

    def _load(self) -> None:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()

    @property
    def dimension(self) -> int:
        return self._DIMS.get(self._model_name, 1536)

    def embed(self, text: str) -> list[float]:
        self._load()
        r = self._client.embeddings.create(input=[text], model=self._model_name)  # type: ignore
        v = np.array(r.data[0].embedding, dtype=np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts: return []
        self._load()
        r = self._client.embeddings.create(input=texts, model=self._model_name)  # type: ignore
        out = []
        for item in r.data:
            v = np.array(item.embedding, dtype=np.float32)
            out.append((v / np.linalg.norm(v)).tolist())
        return out


def create_embedding_engine(config: PersoniqConfig) -> EmbeddingEngine:
    if config.embedding_backend == "openai":
        return OpenAIEmbeddingEngine(model_name=config.embedding_model)
    return LocalEmbeddingEngine(model_name=config.embedding_model)