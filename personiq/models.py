"""
personiq.models
~~~~~~~~~~~~~~~
Core data models. Single source of truth for every layer.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MemoryCategory(str, Enum):
    PREFERENCE = "preference"
    GOAL       = "goal"
    CONTEXT    = "context"
    STYLE      = "style"
    TECHNICAL  = "technical"
    PERSONAL   = "personal"


# Category retrieval boosts — higher = surfaces more often
CATEGORY_WEIGHT: dict[str, float] = {
    MemoryCategory.TECHNICAL:  1.20,
    MemoryCategory.GOAL:       1.15,
    MemoryCategory.PREFERENCE: 1.10,
    MemoryCategory.CONTEXT:    1.05,
    MemoryCategory.PERSONAL:   1.00,
    MemoryCategory.STYLE:      0.90,
}


class Memory(BaseModel):
    id:               str      = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id:          str
    content:          str
    category:         MemoryCategory
    confidence:       float    = Field(default=1.0,  ge=0.0, le=1.0)
    access_count:     int      = Field(default=0,    ge=0)
    decay_factor:     float    = Field(default=1.0,  ge=0.0, le=1.0)
    importance_score: float    = Field(default=1.0,  ge=0.0)
    keywords:         str      = Field(default="")
    source_text:      Optional[str] = None
    created_at:       datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:       datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding:        Optional[list[float]] = Field(default=None, exclude=True)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Memory content must not be empty")
        return v

    def touch(self, confidence_bump: float = 0.05) -> None:
        self.confidence += (1.0 - self.confidence) * confidence_bump
        self.access_count += 1
        self.updated_at   = datetime.now(timezone.utc)
        self._recompute()

    def apply_decay(self, decay_rate: float = 0.005) -> None:
        days              = (datetime.now(timezone.utc) - self.updated_at).total_seconds() / 86400
        self.decay_factor = 1.0 / (1.0 + days * decay_rate)
        self._recompute()

    def _recompute(self) -> None:
        signal               = 1.0 + math.log1p(self.access_count) * 0.1
        self.importance_score = self.confidence * self.decay_factor * signal

    model_config = {"use_enum_values": True}


class ExtractedFact(BaseModel):
    content:    str
    category:   MemoryCategory
    confidence: float      = Field(default=0.9, ge=0.0, le=1.0)
    keywords:   list[str]  = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def strip_it(cls, v: str) -> str:
        return v.strip()


class ExtractionResult(BaseModel):
    facts:        list[ExtractedFact] = Field(default_factory=list)
    raw_response: Optional[str]       = None


class HybridSearchResult(BaseModel):
    memory:          Memory
    semantic_score:  float = Field(default=0.0, ge=-1.0, le=1.0)
    bm25_score:      float = Field(default=0.0, ge=0.0)
    recency_score:   float = Field(default=1.0, ge=0.0, le=1.0)
    category_weight: float = Field(default=1.0, ge=0.0)
    final_score:     float = Field(default=0.0, ge=0.0)


# Backwards-compat alias
class RetrievedMemory(BaseModel):
    memory:     Memory
    similarity: float = Field(ge=0.0, le=1.0)