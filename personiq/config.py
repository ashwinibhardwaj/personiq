"""
personiq.config
~~~~~~~~~~~~~~~
All configuration in one dataclass. Every field has a sane default
and can be overridden via environment variable or constructor arg.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.getenv(f"PERSONIQ_{key}", default)


def _default_db() -> str:
    return str(Path(_env("DB_PATH", "~/.personiq/personiq.db")).expanduser().resolve())


@dataclass
class PersoniqConfig:
    # Storage
    db_path: str = field(default_factory=_default_db)

    # Embeddings
    embedding_backend: str   = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "local"))
    embedding_model:   str   = field(default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # Retrieval
    top_k:                int   = field(default_factory=lambda: int(_env("TOP_K", "5")))
    similarity_threshold: float = field(default_factory=lambda: float(_env("SIMILARITY_THRESHOLD", "0.20")))

    # Hybrid search weights (semantic + bm25 + recency should ≈ 1.0)
    semantic_weight: float = field(default_factory=lambda: float(_env("SEMANTIC_WEIGHT", "0.60")))
    bm25_weight:     float = field(default_factory=lambda: float(_env("BM25_WEIGHT", "0.30")))
    recency_weight:  float = field(default_factory=lambda: float(_env("RECENCY_WEIGHT", "0.10")))

    # Deduplication thresholds
    dedup_threshold: float = field(default_factory=lambda: float(_env("DEDUP_THRESHOLD", "0.85")))
    merge_threshold: float = field(default_factory=lambda: float(_env("MERGE_THRESHOLD", "0.72")))

    # Memory decay
    decay_rate: float = field(default_factory=lambda: float(_env("DECAY_RATE", "0.005")))

    # Extraction
    async_extraction:     bool = field(default_factory=lambda: _env("ASYNC_EXTRACTION", "true").lower() != "false")
    context_window_turns: int  = field(default_factory=lambda: int(_env("CONTEXT_WINDOW", "6")))
    retrieval_queries:    int  = field(default_factory=lambda: int(_env("RETRIEVAL_QUERIES", "3")))

    # Context injection
    max_context_chars: int = 1500

    # Debug
    debug: bool = field(default_factory=lambda: _env("DEBUG", "").lower() in ("1", "true"))

    def __post_init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)